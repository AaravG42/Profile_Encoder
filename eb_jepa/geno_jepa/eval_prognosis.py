"""
Vital Status (Alive/Dead) Prediction using Frozen Encoder + Linear Probe.

Loads a trained GenomicSSL checkpoint, extracts features for all patients that
have vital_status labels in the TCGA-PANCAN clinical dataset, trains a logistic
regression linear probe, and reports accuracy / F1 / precision / recall.
Also produces 2D (PNG) and 3D (HTML) UMAP plots coloured by vital status.

Usage (from eb_jepa/ directory):
    conda activate multiomics_pvae
    python -m geno_jepa.eval_prognosis
    python -m geno_jepa.eval_prognosis --checkpoint /path/to/epoch_75.pth.tar
    python -m geno_jepa.eval_prognosis --device cpu
"""

import json
import os
import sys

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from umap import UMAP
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Paths – override via CLI arguments or edit defaults here
# --------------------------------------------------------------------------- #
_CHECKPOINT = (
    "/home/aarav/Profile_VAE/eb_jepa/checkpoints/image_jepa/"
    "dev_2026-03-09_19-01/"
    "both_mlp_vicreg_proj_pred_bs128_ep150_ph256_po256_pr2048_"
    "std1.0_cov80.0_mlp4096_mask0.3_seed42/epoch_75.pth.tar"
)
_CLINICAL_PATH = "/data/Prognosis_TCGA_PANCAN/clinical_PANCAN_patient_with_followup.tsv"
_GENE_PATH = "/data/TCGA_cleaned/Final_Preprocessed_Gene_Expression_TCGA_CancerTags.pkl"
_METH_PATH = "/data/TCGA_cleaned/Final_Preprocessed_DNA_Methylation_UCSC_PCA_CancerTags.pkl"
_ALIGNMENT_JSON = "/home/aarav/Profile_VAE/patient_alignment_prognosis.json"


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

def _add_repo_to_path():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)


def load_model(checkpoint_path: str, device: torch.device):
    """Initialise GenomicSSL (MLPEncoder backbone) and load checkpoint weights."""
    _add_repo_to_path()
    from geno_jepa.models import MLPEncoder, GenomicSSL

    ck = torch.load(checkpoint_path, map_location=device)
    sd = ck["model_state_dict"]

    # Infer architecture from weight shapes
    in_dim = sd["backbone.mlp.2.weight"].shape[1]   # e.g. 35638
    h1     = sd["backbone.mlp.2.weight"].shape[0]   # e.g. 4096  (hidden_dim)
    feat   = sd["backbone.mlp.10.weight"].shape[0]  # e.g. 1024  (features_dim)
    ph     = sd["projector.0.weight"].shape[0]       # e.g. 256   (proj_hidden_dim)
    po     = sd["projector.6.weight"].shape[0]       # e.g. 256   (proj_output_dim)
    in_channels = 2
    seq_length  = in_dim // in_channels

    print(f"  Inferred: in_channels={in_channels}, seq_length={seq_length}, "
          f"hidden_dim={h1}, features_dim={feat}, "
          f"proj_hidden_dim={ph}, proj_output_dim={po}")

    backbone = MLPEncoder(
        in_channels=in_channels,
        seq_length=seq_length,
        hidden_dim=h1,
    )
    model = GenomicSSL(
        backbone,
        features_dim=feat,
        proj_hidden_dim=ph,
        proj_output_dim=po,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  (Ignored unexpected keys: {unexpected})")
    model.to(device).eval()
    print(f"Loaded model from epoch {ck.get('epoch', '?')}")
    return model, feat


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #

def build_prognosis_tensors(
    gene_path: str,
    meth_path: str,
    clinical_path: str,
    alignment_json: str,
):
    """
    Returns:
        X       : FloatTensor (N, 2, L)  – stacked [gene, meth] per patient
        y       : LongTensor  (N,)       – 0=Alive, 1=Dead
        labels  : list[str]              – ['Alive', 'Dead']
    """
    # --- clinical labels --------------------------------------------------- #
    print("Loading clinical data …")
    clin = pd.read_csv(clinical_path, sep="\t", encoding="latin-1", low_memory=False)
    clin = clin[["bcr_patient_barcode", "vital_status"]].dropna()
    clin["vital_status"] = clin["vital_status"].str.strip()
    clin = clin[clin["vital_status"].isin(["Alive", "Dead"])]
    clinical_map = dict(zip(clin["bcr_patient_barcode"], clin["vital_status"]))

    # --- alignment: full_sample_id -> patient_barcode ---------------------- #
    with open(alignment_json) as fh:
        align = json.load(fh)
    reverse_map: dict[str, str] = {}
    for barcode, full_ids in align["gene_id_mapping"].items():
        for fid in full_ids:
            reverse_map[fid] = barcode

    # --- load omics data --------------------------------------------------- #
    print("Loading gene expression data …")
    df_gene = pd.read_pickle(gene_path)
    df_gene.set_index(df_gene.columns[0], inplace=True)

    print("Loading methylation data …")
    df_meth = pd.read_pickle(meth_path)
    df_meth.set_index(df_meth.columns[0], inplace=True)

    # Align samples & genes (same logic as GenomicDataset)
    common_samples = df_gene.columns.intersection(df_meth.columns)
    common_genes   = (
        df_gene.index.intersection(df_meth.index)
        .drop("Cancer_Type", errors="ignore")
    )
    print(f"Common samples: {len(common_samples)}, common genes: {len(common_genes)}")

    # --- build label vector ------------------------------------------------ #
    vital_labels: list[int] = []
    valid_samples: list[str] = []

    for sample in common_samples:
        # Lookup barcode: use pre-built reverse map, fall back to truncation
        barcode = reverse_map.get(sample)
        if barcode is None and len(sample) > 12 and sample[-3] == "-":
            barcode = sample[:-3]
        if barcode is None or barcode not in clinical_map:
            continue
        vital = clinical_map[barcode]
        vital_labels.append(0 if vital == "Alive" else 1)
        valid_samples.append(sample)

    n_alive = vital_labels.count(0)
    n_dead  = vital_labels.count(1)
    print(f"Patients with vital_status label: {len(valid_samples)} "
          f"(Alive={n_alive}, Dead={n_dead})")

    # --- extract numeric arrays for valid samples -------------------------- #
    gene_arr = df_gene.loc[common_genes, valid_samples].values.astype(np.float32).T
    meth_arr = df_meth.loc[common_genes, valid_samples].values.astype(np.float32).T

    gene_t = torch.from_numpy(gene_arr)  # (N, L)
    meth_t = torch.from_numpy(meth_arr)  # (N, L)
    X = torch.stack([gene_t, meth_t], dim=1)  # (N, 2, L)
    y = torch.tensor(vital_labels, dtype=torch.long)

    return X, y, ["Alive", "Dead"]


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def extract_features(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats_list = []
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Extracting features"):
            batch = batch.to(device)
            feats, _ = model(batch)
            feats_list.append(feats.cpu().numpy())
    return np.concatenate(feats_list, axis=0)


# --------------------------------------------------------------------------- #
# Linear probe training & evaluation
# --------------------------------------------------------------------------- #

def train_and_eval_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    test_size: float = 0.2,
    seed: int = 42,
):
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    print("Training logistic regression …")
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed, n_jobs=-1)
    clf.fit(X_tr_s, y_tr)

    y_pred = clf.predict(X_te_s)

    acc  = accuracy_score(y_te, y_pred)
    # report class 1 = Dead
    f1   = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_te, y_pred, pos_label=1, zero_division=0)

    print("\n" + "=" * 60)
    print("  VITAL STATUS PREDICTION — Linear Probe Results")
    print("=" * 60)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  F1 (Dead) : {f1:.4f}")
    print(f"  Precision : {prec:.4f}  (Dead class)")
    print(f"  Recall    : {rec:.4f}  (Dead class)")
    print("\nFull classification report:")
    print(classification_report(y_te, y_pred, target_names=label_names, zero_division=0))

    return acc, f1, prec, rec, y_te, y_pred


# --------------------------------------------------------------------------- #
# UMAP visualisations
# --------------------------------------------------------------------------- #

_COLORS = {0: "#2196F3", 1: "#F44336"}  # Blue=Alive, Red=Dead


def plot_umap_2d(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    save_path: str,
):
    print("Computing 2D UMAP …")
    emb = UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="euclidean", random_state=42,
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, name in enumerate(label_names):
        mask = labels == cls_idx
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=_COLORS[cls_idx], label=name,
            alpha=0.65, s=10, linewidths=0,
        )
    ax.set_title("Latent Space (2D UMAP) — Vital Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=5, fontsize=12, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D UMAP → {save_path}")
    return emb


def plot_umap_3d(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    save_path: str,
):
    print("Computing 3D UMAP …")
    emb = UMAP(
        n_neighbors=15, min_dist=0.1, n_components=3,
        metric="euclidean", random_state=42,
    ).fit_transform(features)

    traces = []
    for cls_idx, name in enumerate(label_names):
        mask = labels == cls_idx
        traces.append(
            go.Scatter3d(
                x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
                mode="markers",
                name=name,
                marker=dict(size=3, color=_COLORS[cls_idx], opacity=0.75),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Latent Space (3D UMAP) — Vital Status",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        legend=dict(itemsizing="constant", font=dict(size=14)),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    fig.write_html(save_path)
    print(f"Saved 3D UMAP → {save_path}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main(
    checkpoint: str = _CHECKPOINT,
    clinical_path: str = _CLINICAL_PATH,
    gene_path: str = _GENE_PATH,
    meth_path: str = _METH_PATH,
    alignment_json: str = _ALIGNMENT_JSON,
    device: str = "cuda",
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
    out_dir: str = None,
):
    """
    Args:
        checkpoint:     Path to .pth.tar checkpoint file.
        clinical_path:  Path to TCGA-PANCAN clinical TSV.
        gene_path:      Path to gene expression .pkl file.
        meth_path:      Path to methylation .pkl file.
        alignment_json: Path to patient alignment JSON.
        device:         'cuda' or 'cpu'.
        batch_size:     Batch size for feature extraction.
        test_size:      Fraction of data used for testing (0–1).
        seed:           Random seed.
        out_dir:        Directory to write output files (defaults to checkpoint directory).
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    if out_dir is None:
        out_dir = os.path.dirname(checkpoint)
        print(f"Using default output directory: {out_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Load model
    print("\n[1/4] Loading model …")
    model, _ = load_model(checkpoint, dev)

    # 2. Build prognosis dataset
    print("\n[2/4] Building prognosis dataset …")
    X, y, label_names = build_prognosis_tensors(
        gene_path, meth_path, clinical_path, alignment_json
    )

    # 3. Extract encoder features
    print("\n[3/4] Extracting encoder features …")
    features = extract_features(model, X, dev, batch_size=batch_size)
    labels   = y.numpy()
    print(f"Features shape: {features.shape}")

    # 4. Train linear probe & report metrics
    print("\n[4/4] Training and evaluating linear probe …")
    train_and_eval_linear_probe(features, labels, label_names, test_size=test_size, seed=seed)

    # 5. UMAP plots
    plot_umap_2d(
        features, labels, label_names,
        save_path=os.path.join(out_dir, "vital_status_umap2d.png"),
    )
    plot_umap_3d(
        features, labels, label_names,
        save_path=os.path.join(out_dir, "vital_status_umap3d.html"),
    )

    print("\nDone.")


if __name__ == "__main__":
    fire.Fire(main)

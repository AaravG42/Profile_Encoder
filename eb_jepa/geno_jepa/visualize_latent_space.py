"""
Visualize the latent space of a trained Genomic JEPA model using UMAP.

Usage:
    python -m geno_jepa.visualize_latent_space \
        --checkpoint_path /path/to/checkpoint.pth.tar \
        --model_type conv1d \
        --use_channels both \
"""

import os
import pickle
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import VisionTransformer
from tqdm import tqdm
from umap import UMAP

from geno_jepa.dataset import GenomicDataset, get_genomic_val_transforms
from geno_jepa.main import GenomicSSL, ResNet18, Conv1DEncoder, ViT1DEncoder, MLPEncoder


def load_model_from_checkpoint(
    checkpoint_path,
    device="cuda",
    model_type=None,
    patch_size=100,
    use_channels="both",
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    mlp_dim=512,
):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to infer model type from checkpoint path if not provided
    if model_type is None:
        checkpoint_path_str = str(checkpoint_path)
        if "conv1d" in checkpoint_path_str.lower():
            model_type = "conv1d"
        elif "vit" in checkpoint_path_str.lower():
            model_type = "vit"
        elif "mlp" in checkpoint_path_str.lower():
            model_type = "mlp"
        else:
            model_type = "resnet"
    
    print(f"Using model type: {model_type}")
    in_channels = 2 if use_channels == "both" else 1
    
    # Initialize backbone
    if model_type == "conv1d":
        backbone = Conv1DEncoder(in_channels=in_channels)
        features_dim = backbone.features_dim
        print(f"Using Conv1DEncoder backbone (features_dim={features_dim})")
    elif model_type == "resnet":
        backbone = ResNet18(in_channels=in_channels)
        features_dim = backbone.features_dim
        print(f"Using ResNet18 backbone with {in_channels} input channels")
    elif model_type == "vit":
        backbone = ViT1DEncoder(
            in_channels=in_channels,
            seq_length=15703,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
        )
        features_dim = backbone.features_dim
        print(f"Using ViT1DEncoder backbone (patch_size={patch_size}, features_dim={features_dim})")
    elif model_type == "mlp":
        backbone = MLPEncoder(
            in_channels=in_channels,
            seq_length=15703,
            hidden_dim=hidden_dim,
        )
        features_dim = backbone.features_dim
        print(f"Using MLPEncoder backbone (hidden_dim={hidden_dim}, features_dim={features_dim})")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize model
    model = GenomicSSL(
        backbone,
        features_dim=features_dim,
        proj_hidden_dim=2048,
        proj_output_dim=2048,
    )
    
    # Load state dict
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Model loaded from epoch: {epoch}")
    
    return model


def load_genomic_dataset(base_path, patch_size=None, use_channels="both"):
    """Load the genomic dataset."""
    methylation_path = os.path.join(base_path, "methylation_tensor_chrom_ordered.pkl")
    gene_expression_path = os.path.join(base_path, "gene_expression_tensor_chrom_ordered.pkl")
    labels_path = os.path.join(base_path, "cancer_tags_tensor_chrom_ordered.pkl")
    
    dataset = GenomicDataset(
        methylation_path=methylation_path,
        gene_expression_path=gene_expression_path,
        labels_path=labels_path,
        transform=get_genomic_val_transforms(),  # No augmentation for visualization
        patch_size=patch_size,
        use_channels=use_channels,
    )
    
    return dataset


def extract_latent_representations(model, dataset, device="cuda", batch_size=32):
    """Extract latent representations from the encoder."""
    print("Extracting latent representations...")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for views, labels in tqdm(dataloader, desc="Extracting features"):
            # Handle views as list
            if isinstance(views, list):
                data = views[0]
            else:
                data = views
            
            data = data.to(device)
            
            # Extract features from encoder (not projections)
            features, _ = model(data)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Extracted features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Number of unique classes: {len(np.unique(all_labels))}")
    
    return all_features, all_labels


def get_cancer_type_names():
    """Get cancer type names mapping from the dataset."""
    # Try to load from cancer_class_mapping.json
    import json
    
    # Try multiple possible locations
    possible_paths = [
        "/home/dmlab/Devendra/cancer_class_mapping.json",
        "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/cancer_class_mapping.json",
        "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/cancer_class_mapping.json",
    ]
    
    # for path in possible_paths:
    #     if os.path.exists(path):
    #         print(f"Loading cancer type mapping from: {path}")
    #         with open(path, "r") as f:
    #             mapping = json.load(f)
    #         # Convert string keys to integers
    #         return {int(k): v for k, v in mapping.items()}
    
    # Fallback: Use the known TCGA cancer types mapping
    print("Warning: cancer_class_mapping.json not found, using default TCGA mapping")
    # cancer_mapping = {
    #     0: "ACC",
    #     1: "BLCA",
    #     2: "BRCA",
    #     3: "CESC",
    #     4: "CHOL",
    #     5: "COAD",
    #     6: "DLBC",
    #     7: "ESCA",
    #     8: "GBM",
    #     9: "HNSC",
    #     10: "KICH",
    #     11: "KIRC",
    #     12: "KIRP",
    #     13: "LAML",
    #     14: "LGG",
    #     15: "LIHC",
    #     16: "LUAD",
    #     17: "LUSC",
    #     18: "MESO",
    #     19: "OV",
    #     20: "PAAD",
    #     21: "PCPG",
    #     22: "PRAD",
    #     23: "READ",
    #     24: "SARC",
    #     25: "SKCM",
    #     26: "STAD",
    #     27: "TGCT",
    # }
    cancer_mapping = {
        0: "ACC",
        1: "BLCA",
        2: "BRCA",
        3: "CESC",
        4: "COAD",
        5: "ESCA",
        6: "GBM",
        7: "HNSC",
        8: "KICH",
        9: "KIRC",
        10: "KIRP",
        11: "LGG",
        12: "LIHC",
        13: "LUAD",
        14: "LUSC",
        15: "MESO",
        16: "OV",
        17: "PAAD",
        18: "PCPG",
        19: "PRAD",
        20: "READ",
        21: "SARC",
        22: "SKCM",
        23: "STAD",
        24: "TGCT",
        25: "THCA",
        26: "UCEC",
        27: "UCS"
    }
    return cancer_mapping


def plot_umap(features, labels, save_path="latent_space_umap.png", title="Latent Space UMAP"):
    """Create UMAP visualization of latent space."""
    print("Computing UMAP projection...")
    
    # Apply UMAP
    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    
    embedding = reducer.fit_transform(features)
    
    print(f"UMAP embedding shape: {embedding.shape}")
    
    # Get cancer type names
    cancer_mapping = get_cancer_type_names()
    unique_labels = np.unique(labels)
    
    # Create color map
    cmap = plt.cm.get_cmap("tab20")
    if len(unique_labels) > 20:
        cmap = plt.cm.get_cmap("tab20b")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each cancer type
    for idx, label in enumerate(sorted(unique_labels)):
        mask = labels == label
        cancer_name = cancer_mapping.get(label, f"Class {label}")
        
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[cmap(idx / len(unique_labels))],
            label=cancer_name,
            alpha=0.6,
            s=20,
            edgecolors="none",
        )
    
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Create legend with multiple columns if many classes
    ncol = 2 if len(unique_labels) > 10 else 1
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=ncol,
        fontsize=9,
        markerscale=1.5,
        framealpha=0.9,
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"UMAP visualization saved to: {save_path}")
    
    # Also save a high-res version
    high_res_path = save_path.parent / f"{save_path.stem}_highres.png"
    plt.savefig(high_res_path, dpi=600, bbox_inches="tight")
    print(f"High-res version saved to: {high_res_path}")
    
    plt.close()
    
    # Create a version with density coloring
    plot_umap_density(embedding, labels, save_path.parent / f"{save_path.stem}_density.png")


def plot_umap_density(embedding, labels, save_path):
    """Create a density-colored UMAP visualization."""
    from scipy.stats import gaussian_kde
    
    print("Creating density-colored UMAP...")
    
    # Calculate point density
    xy = embedding.T
    z = gaussian_kde(xy)(xy)
    
    # Sort points by density, so densest points are plotted last
    idx = z.argsort()
    x, y, z = embedding[idx, 0], embedding[idx, 1], z[idx]
    labels_sorted = labels[idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        x, y,
        c=z,
        s=20,
        alpha=0.5,
        cmap="viridis",
        edgecolors="none",
    )
    
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.set_title("Latent Space UMAP (Density Colored)", fontsize=14, fontweight="bold")
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Point Density", fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Density UMAP saved to: {save_path}")
    plt.close()


def main(
    checkpoint_path: str = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/Aarav_exps/eb_jepa/checkpoints/image_jepa/dev_2026-02-10_05-01/conv1d_vicreg_proj_bs32_ep150_ph2048_po2048_std1.0_cov80.0_seed42/latest.pth.tar",
    data_path: str = "/home/aarav/data/chromosome_coordinate",
    save_path: str = None,
    device: str = "cuda",
    batch_size: int = 32,
    model_type: str = None,
    patch_size: int = 100,
    use_channels: str = "both",
    hidden_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    mlp_dim: int = 512,
):
    """
    Visualize latent space of trained Genomic JEPA model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to genomic data directory
        save_path: Path to save visualization (default: next to checkpoint)
        device: Device to use (cuda/cpu)
        batch_size: Batch size for feature extraction
        model_type: Force model type ('conv1d', 'vit', 'mlp', 'resnet')
        patch_size: Patch size for ViT or data reshaping
        use_channels: Which channels to use ('both', 'gene', or 'meth')
        hidden_dim: Hidden dimension for ViT
        num_layers: Number of layers for ViT
        num_heads: Number of heads for ViT
        mlp_dim: MLP dimension for ViT
    """
    # Auto-detect save path if not provided
    if save_path is None:
        checkpoint_dir = Path(checkpoint_path).parent
        epoch = Path(checkpoint_path).stem.replace("epoch_", "ep")
        save_path = checkpoint_dir / f"latent_space_umap_{epoch}.png"
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path,
        device=device,
        model_type=model_type,
        patch_size=patch_size,
        use_channels=use_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
    )
    
    # Load dataset
    dataset = load_genomic_dataset(data_path, patch_size=patch_size, use_channels=use_channels)
    
    # Extract latent representations
    features, labels = extract_latent_representations(
        model, dataset, device=device, batch_size=batch_size
    )
    
    # Save features and labels for future use
    features_save_path = Path(save_path).parent / "latent_features.npz"
    np.savez(features_save_path, features=features, labels=labels)
    print(f"Features saved to: {features_save_path}")
    
    # Create UMAP visualization
    checkpoint_name = Path(checkpoint_path).parent.name
    epoch_name = Path(checkpoint_path).stem
    title = f"Latent Space UMAP - {checkpoint_name}\n({epoch_name})"
    
    plot_umap(features, labels, save_path=save_path, title=title)
    
    print("\nVisualization complete!")
    print(f"Main plot: {save_path}")
    print(f"Features: {features_save_path}")


if __name__ == "__main__":
    fire.Fire(main)

# python -m Genotype_Induced_Drug_Design.PVAE.Aarav_exps.mse_vit_vae_train_script

import os
import pickle
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from Genotype_Induced_Drug_Design.PVAE.Aarav_exps.vit_vae import ViTVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders_supervised

# Optional wandb import
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# _HAS_WANDB = False

BEST_LAMB = 0.1212279236206466
BEST_ALPHA = 13.780485764196705
BEST_MASK = 0.20973865433962333
BEST_BLOCK = 300
BEST_ZDIM = 64
PATCH_SIZE = 16


def build_model(input_dim: int, z_dim: int, num_classes: int):
    model = ViTVAE(
        input_dim=input_dim,
        z_dim=z_dim,
        num_classes=num_classes,
        patch_size=PATCH_SIZE,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
    )
    return model


def augment_with_gaussian_noise(dna, gene, labels, noise_std=0.05):
    print(f"\n--- Augmentation Started (std={noise_std}) ---")
    print(f"Original shape: {dna.shape}")

    noise_dna = torch.randn_like(dna) * noise_std
    noise_gene = torch.randn_like(gene) * noise_std

    aug_dna = dna + noise_dna
    aug_gene = gene + noise_gene

    final_dna = torch.cat([dna, aug_dna], dim=0)
    final_gene = torch.cat([gene, aug_gene], dim=0)
    final_labels = torch.cat([labels, labels], dim=0)

    print(f"Augmented shape: {final_dna.shape}")
    print("--- Augmentation Completed ---\n")

    return final_dna, final_gene, final_labels


@torch.no_grad()
def evaluate(model, dataloader, device, lamb=1.0, alpha=20.0):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for x_dna_meth, x_gene_exp, labels in dataloader:
        x_dna_meth = x_dna_meth.to(device)
        x_gene_exp = x_gene_exp.to(device)
        labels = labels.to(device)

        recon_dna, recon_gene, mu, logvar, class_logits = model.forward_with_classifier(x_dna_meth, x_gene_exp)

        loss, _, _, _, cls_loss = model.loss(
            x_dna=x_dna_meth,
            x_gene=x_gene_exp,
            r_dna=recon_dna,
            r_gene=recon_gene,
            mu=mu,
            logvar=logvar,
            labels=labels,
            preds=class_logits,
            lamb=lamb,
            alpha=alpha,
        )

        preds_cls = torch.argmax(class_logits, dim=1)
        acc = (preds_cls == labels.view(-1)).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    return avg_loss, avg_acc


def main():
    input_dim = 15703

    gaussian_aug = True
    aug_noise_std = 0.1

    # --- Data Loading (chromosome-ordered files preserved) ---
    with open(
        "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/chromosome_coordinate/methylation_tensor_chrom_ordered.pkl",
        "rb",
    ) as f:
        dna_meth = pickle.load(f)

    with open(
        "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/chromosome_coordinate/gene_expression_tensor_chrom_ordered.pkl",
        "rb",
    ) as f:
        gene_exp = pickle.load(f)

    try:
        with open(
            "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/chromosome_coordinate/cancer_tags_tensor_chrom_ordered.pkl",
            "rb",
        ) as f:
            labels = pickle.load(f)
    except FileNotFoundError:
        print("Labels file not found, creating dummy labels.")
        labels = torch.randint(0, 2, (len(dna_meth),))

    # --- Label processing ---
    if labels.dim() > 1 and labels.shape[1] > 1:
        print(f"Detected One-Hot Labels with shape {labels.shape}. Converting to indices...")
        num_classes = labels.shape[1]
        labels = torch.argmax(labels, dim=1)
    else:
        num_classes = len(torch.unique(labels))

    print(f"Final detected classes: {num_classes}")

    dna_meth = dna_meth.to(dtype=torch.float32)
    gene_exp = gene_exp.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.long)

    train_loader, val_loader, test_loader = return_dataloaders_supervised(
        dna_meth, gene_exp, labels, split_fractions=(0.8, 0.1)
    )

    # --- Augment only training split ---
    if gaussian_aug:
        try:
            idx = train_loader.dataset.indices
            x_train_dna = dna_meth[idx]
            x_train_gene = gene_exp[idx]
            y_train = labels[idx]

            x_train_dna, x_train_gene, y_train = augment_with_gaussian_noise(
                x_train_dna, x_train_gene, y_train, noise_std=aug_noise_std
            )

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_train_dna, x_train_gene, y_train),
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=getattr(train_loader, "num_workers", 0),
                drop_last=False,
            )
        except Exception:
            # Fallback: if dataset doesn't expose indices, augment entire set (preserve behavior)
            dna_meth, gene_exp, labels = augment_with_gaussian_noise(dna_meth, gene_exp, labels, noise_std=aug_noise_std)
            train_loader, val_loader, test_loader = return_dataloaders_supervised(dna_meth, gene_exp, labels, split_fractions=(0.8, 0.1))

    print(f"Train batches: {len(train_loader)}")

    # --- Model / Optimizer ---
    model = build_model(input_dim=input_dim, z_dim=128, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # --- wandb init ---
    if _HAS_WANDB:
        run = wandb.init(
            project="Genotype_ViTVAE",
            config={
                "input_dim": input_dim,
                "z_dim": 128,
                "patch_size": PATCH_SIZE,
                "embed_dim": 256,
                "num_layers": 6,
                "num_heads": 8,
                "lr": 1e-4,
                "batch_size": getattr(train_loader, 'batch_size', 16),
            },
        )
        wandb.watch(model, log="all", log_freq=10)
    else:
        run = None

    # --- Training using model.trainer (keeps same print statements) ---
    history, mu_logvar_history, test_history = model.trainer(
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=150,
        device=device,
        lamb=BEST_LAMB,
        alpha=BEST_ALPHA,
        log_interval=1,
        patience=10,
        verbose=True,
        test_loader=test_loader,
        apply_masking=False,
        mask_ratio=BEST_MASK,
        block_size=BEST_BLOCK,
    )

    # Scheduler step if desired
    try:
        scheduler.step()
    except Exception:
        pass

    # --- Evaluation ---
    train_loss, train_acc = evaluate(model, train_loader, device, lamb=BEST_LAMB, alpha=BEST_ALPHA)
    val_loss, val_acc = evaluate(model, val_loader, device, lamb=BEST_LAMB, alpha=BEST_ALPHA)
    test_loss, test_acc = evaluate(model, test_loader, device, lamb=BEST_LAMB, alpha=BEST_ALPHA)

    # --- Save model and histories ---
    save_dir = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/Aarav_exps"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vit_vae_supervised_model_coordinate_patch{PATCH_SIZE}.pt")
    model.save_model(model, save_path)
    print(f"Model saved to {save_path}")

    hist_path = os.path.join(save_dir, f"vit_vae_supervised_history_coordinate_patch{PATCH_SIZE}.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump({"train_history": history, "mu_logvar_history": mu_logvar_history, "test_history": test_history}, f)
    print("Histories saved.")

    # --- Print final results (preserve behavior) ---
    print(f"\nFinal Results:")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # --- wandb final logging ---
    if _HAS_WANDB:
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })
        wandb.save(save_path)
        wandb.save(hist_path)
        wandb.finish()

    return history, mu_logvar_history, test_history, train_loss, val_loss, test_loss


if __name__ == "__main__":
    main()

# python -m Genotype_Induced_Drug_Design.PVAE.Aarav_exps.ijepa_train_script

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Genotype_Induced_Drug_Design.PVAE.Aarav_exps.ijepa_model import (
    GenomicIJEPA, MaskGenerator, momentum_schedule
)
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders_supervised

# Optional wandb import
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def ijepa_loss(predictions, targets):
    """
    Smooth L1 loss between predicted and target representations.
    Both inputs are assumed to be layer-normalized.
    """
    # Apply layer normalization
    predictions = F.layer_norm(predictions, (predictions.size(-1),))
    targets = F.layer_norm(targets, (targets.size(-1),))
    
    # Smooth L1 loss (Huber loss)
    loss = F.smooth_l1_loss(predictions, targets, reduction='mean')
    return loss


def extract_target_features(full_features, target_mask):
    """
    Extract features at target positions from full feature map.
    full_features: (B, n_patches, embed_dim)
    target_mask: (B, n_patches) binary mask
    Returns: (B, n_targets, embed_dim)
    """
    B, N, D = full_features.shape
    
    # Get maximum number of targets in batch
    n_targets = int(target_mask.sum(dim=1).max().item())
    
    # Extract target features for each sample
    target_features = []
    for i in range(B):
        mask_i = target_mask[i].bool()
        feats_i = full_features[i][mask_i]  # (n_targets_i, D)
        
        # Pad if needed
        if feats_i.shape[0] < n_targets:
            pad = torch.zeros(n_targets - feats_i.shape[0], D, device=feats_i.device)
            feats_i = torch.cat([feats_i, pad], dim=0)
        
        target_features.append(feats_i[:n_targets])
    
    return torch.stack(target_features)


@torch.no_grad()
def evaluate_classification(model, dataloader, device):
    """Evaluate classification accuracy using context encoder."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for x_dna, x_gene, labels in dataloader:
        x_dna, x_gene, labels = x_dna.to(device), x_gene.to(device), labels.to(device)
        
        logits = model.forward_classifier(x_dna, x_gene)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels.view(-1)).sum().item()
        total_samples += labels.size(0)
    
    accuracy = total_correct / max(total_samples, 1)
    return accuracy


def train_ijepa(
    model,
    train_loader,
    test_loader,
    optimizer,
    mask_generator,
    device,
    num_epochs=150,
    ema_momentum=(0.996, 0.9999),
    log_interval=50,
    verbose=True,
    wandb_run=None
):
    """
    I-JEPA training loop with EMA updates and optional wandb logging.
    """
    model.to(device)
    
    # Generate momentum schedule
    momentum_values = momentum_schedule(ema_momentum[0], ema_momentum[1], num_epochs)
    
    history = {
        'train_loss': [],
        'test_acc': [],
        'momentum': []
    }
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        current_momentum = momentum_values[epoch - 1]
        
        for batch_idx, (x_dna, x_gene, labels) in enumerate(train_loader):
            x_dna, x_gene = x_dna.to(device), x_gene.to(device)
            labels = labels.to(device)
            
            batch_size = x_dna.size(0)
            
            # Generate masks
            context_mask, target_mask = mask_generator.generate_masks(batch_size, device)
            
            # 1. Target encoder forward (no gradients)
            target_features = model.forward_target(x_dna, x_gene)
            # Extract target positions
            h = extract_target_features(target_features, target_mask)
            
            # 2. Context encoder forward
            z_context = model.forward_context(x_dna, x_gene, context_mask)
            
            # 3. Predictor forward
            z_pred = model.forward_predictor(z_context, context_mask, target_mask)
            
            # 4. Compute loss
            loss = ijepa_loss(z_pred, h)
            
            # 5. Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 6. Update target encoder (EMA)
            model.update_target_encoder(current_momentum)
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Batch logging
            if verbose and (batch_idx + 1) % log_interval == 0:
                avg_loss = loss.item()
                n_context = context_mask.sum(dim=1).float().mean().item()
                n_target = target_mask.sum(dim=1).float().mean().item()
                print(
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f} | "
                    f"Context: {n_context:.1f} | Target: {n_target:.1f} patches"
                )
            
            if wandb_run:
                wandb_run.log({
                    'batch/loss': loss.item(),
                    'batch/context_patches': context_mask.sum(dim=1).float().mean().item(),
                    'batch/target_patches': target_mask.sum(dim=1).float().mean().item(),
                })
        
        # Epoch metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        history['momentum'].append(current_momentum)
        
        # Evaluate classification accuracy
        test_acc = evaluate_classification(model, test_loader, device)
        history['test_acc'].append(test_acc)
        
        if verbose:
            print(
                f"\nEpoch [{epoch}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Test Acc: {test_acc:.4f} | "
                f"Momentum: {current_momentum:.5f}\n"
            )
        
        if wandb_run:
            wandb_run.log({
                'epoch': epoch,
                'train/loss': avg_train_loss,
                'test/accuracy': test_acc,
                'ema_momentum': current_momentum,
            })
    
    return history


def main():
    input_dim = 15703
    patch_size = 128
    
    # --- Data Loading (chromosome-ordered files) ---
    print("Loading data...")
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
    print(f"Data shapes: DNA {dna_meth.shape}, Gene {gene_exp.shape}, Labels {labels.shape}")
    
    dna_meth = dna_meth.to(dtype=torch.float32)
    gene_exp = gene_exp.to(dtype=torch.float32)
    labels = labels.to(dtype=torch.long)
    
    # --- Create dataloaders ---
    train_loader, val_loader, test_loader = return_dataloaders_supervised(
        dna_meth, gene_exp, labels, split_fractions=(0.8, 0.1)
    )
    
    # Rebuild with smaller batch size to avoid OOM
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)} (Batch size: {batch_size})")
    
    # --- Model initialization ---
    n_patches = (input_dim + patch_size - 1) // patch_size
    print(f"Number of patches: {n_patches}")
    
    model = GenomicIJEPA(
        input_dim=input_dim,
        patch_size=patch_size,
        embed_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_heads=6,
        dropout=0.1,
        num_classes=num_classes
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Mask generator ---
    mask_generator = MaskGenerator(
        n_patches=n_patches,
        num_context_blocks=1,
        num_target_blocks=4,
        context_scale=(0.85, 1.0),
        target_scale=(0.15, 0.2),
        min_keep=10,
        allow_overlap=False
    )
    
    # --- Optimizer ---
    optimizer = optim.AdamW(
        [
            {'params': model.context_encoder.parameters()},
            {'params': model.predictor.parameters()},
        ],
        lr=1e-4,
        weight_decay=0.05
    )
    
    # --- wandb init ---
    wandb_run = None
    if _HAS_WANDB:
        wandb_run = wandb.init(
            project="Genotype_IJEPA",
            config={
                "architecture": "I-JEPA",
                "input_dim": input_dim,
                "patch_size": patch_size,
                "embed_dim": 768,
                "encoder_depth": 12,
                "predictor_depth": 6,
                "num_patches": n_patches,
                "context_blocks": 1,
                "target_blocks": 4,
                "lr": 1e-4,
                "weight_decay": 0.05,
                "batch_size": getattr(train_loader, 'batch_size', 64),
            },
        )
        wandb.watch(model.context_encoder, log="gradients", log_freq=100)
    
    # --- Training ---
    print("\nStarting I-JEPA training...")
    history = train_ijepa(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        mask_generator=mask_generator,
        device=device,
        num_epochs=150,
        ema_momentum=(0.996, 0.9999),
        log_interval=50,
        verbose=True,
        wandb_run=wandb_run
    )
    
    # --- Final evaluation ---
    print("\n=== Final Evaluation ===")
    train_acc = evaluate_classification(model, train_loader, device)
    val_acc = evaluate_classification(model, val_loader, device)
    test_acc = evaluate_classification(model, test_loader, device)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # --- Save model ---
    save_dir = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/Aarav_exps"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "ijepa_model_coordinate.pt")
    torch.save({
        'context_encoder': model.context_encoder.state_dict(),
        'target_encoder': model.target_encoder.state_dict(),
        'predictor': model.predictor.state_dict(),
        'classifier': model.classifier.state_dict(),
    }, save_path)
    print(f"\nModel saved to {save_path}")
    
    # Save history
    hist_path = os.path.join(save_dir, "ijepa_history_coordinate.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    print(f"History saved to {hist_path}")
    
    # --- wandb final logging ---
    if _HAS_WANDB:
        wandb.log({
            "final/train_acc": train_acc,
            "final/val_acc": val_acc,
            "final/test_acc": test_acc,
        })
        wandb.save(save_path)
        wandb.save(hist_path)
        wandb.finish()
    
    return history, train_acc, val_acc, test_acc


if __name__ == "__main__":
    main()

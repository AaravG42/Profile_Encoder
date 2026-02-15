"""
Genomic VICReg Training Script - Native PyTorch Implementation

This script implements VICReg training on genomic dataset using only PyTorch.
Supports 1D Convolutional encoder for genomic data, and ResNet/ViT backbones for CIFAR-10.

Usage:
    # With YAML config:
    python -m geno_jepa.main --fname geno_jepa/cfgs/genomic.yaml

    # With config + overrides:
    python -m geno_jepa.main --fname geno_jepa/cfgs/genomic.yaml optim.epochs=50
"""

import os
import time
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim.optimizer import required
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import VisionTransformer
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.losses import BCS, VICRegLoss
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from geno_jepa.dataset import (
    ImageDataset,
    get_train_transforms,
    get_val_transforms,
    GenomicDataset,
    get_genomic_train_transforms,
    get_genomic_val_transforms,
)
from geno_jepa.eval import LinearProbe, evaluate_linear_probe

logger = get_logger(__name__)


class ResNet18(nn.Module):
    """ResNet-18 backbone implementation."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.backbone = torchvision.models.resnet18()
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.features_dim = 512

    def forward(self, x):
        return self.backbone(x)


class Conv1DEncoder(nn.Module):
    """1D Convolutional Encoder for genomic data."""

    def __init__(self, in_channels=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16), nn.LeakyReLU(),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.LeakyReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(),

            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
        )
        # Calculate output size: 15703 -> 7852 -> 3926 -> 1963 -> 982 -> 491
        # Output: (batch_size, 64, 491)
        self.features_dim = 64 * 491  # Flattened feature dimension
        
    def forward(self, x):
        # x shape can be (batch_size, C, L) or (batch_size, C, N, P)
        if x.dim() == 4:
            # Flatten patches back to 1D if it was reshaped in dataset
            b, c, n, p = x.shape
            x = x.reshape(b, c, n * p)
            # Clip if it was padded beyond 15703 (though Conv1d is usually fine)
            x = x[:, :, :15703]
            
        x = self.encoder(x)
        # x shape: (batch_size, 64, 491)
        x = x.flatten(start_dim=1)  # Flatten to (batch_size, 64*491)
        return x


class MLPEncoder(nn.Module):
    """Simple MLP Encoder for genomic data."""

    def __init__(self, in_channels=2, seq_length=15703, hidden_dim=1024, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        input_dim = in_channels * seq_length
        
        # Simple 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
        )
        self.features_dim = hidden_dim // 4
        
    def forward(self, x):
        # Flatten to (batch_size, C * L)
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x


class ViT1DEncoder(nn.Module):
    """1D Vision Transformer Encoder for genomic data."""

    def __init__(
        self,
        in_channels=2,
        seq_length=15703,
        patch_size=100,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Calculate number of patches
        self.num_patches = (seq_length + patch_size - 1) // patch_size

        # Patch projection: using Linear layer since input is (batch, C, N, P)
        self.patch_embed = nn.Linear(in_channels * patch_size, hidden_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.features_dim = hidden_dim

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x shape: (batch_size, C, N, P)
        b, c, n, p = x.shape
        
        # Reshape to (batch_size, N, C*P)
        x = x.permute(0, 2, 1, 3).reshape(b, n, c * p)
        
        x = self.patch_embed(x)  # (batch_size, num_patches, hidden_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, hidden_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Return CLS token feature
        return x[:, 0]


class GenomicSSL(nn.Module):
    """Genomic Self-Supervised Learning model implementation."""

    def __init__(
        self, backbone, features_dim, proj_hidden_dim=2048, proj_output_dim=2048
    ):
        super().__init__()
        self.backbone = backbone
        self.features_dim = features_dim

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


class LARS(optim.Optimizer):
    """LARS optimizer implementation."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        eta=1e-3,
        eps=1e-8,
        clip_lr=False,
        exclude_bias_n_norm=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta,
            eps=eps,
            clip_lr=clip_lr,
            exclude_bias_n_norm=exclude_bias_n_norm,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if p.ndim != 1 or not group["exclude_bias_n_norm"]:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (
                            g_norm + p_norm * weight_decay + group["eps"]
                        )
                        lars_lr *= group["eta"]

                        # clip lr
                        if group["clip_lr"]:
                            lars_lr = min(lars_lr / group["lr"], 1)

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


class WarmupCosineScheduler:
    """Warmup cosine learning rate scheduler"""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        base_lr,
        min_lr=0.0,
        warmup_start_lr=3e-5,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + epoch * (
                self.base_lr - self.warmup_start_lr
            ) / (self.warmup_epochs - 1)
        else:
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1
                + torch.cos(
                    torch.tensor(
                        (epoch - self.warmup_epochs)
                        / (self.max_epochs - self.warmup_epochs)
                        * 3.14159
                    )
                )
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    linear_probe,
    scaler,
    device,
    epoch,
    loss_fn,
    use_amp=True,
    dtype=torch.float16,
    tqdm_silent=False,
):
    """Train for one epoch."""
    model.train()
    linear_probe.train()

    # Dynamic loss accumulator
    loss_totals = {}
    total_linear_loss = 0
    linear_correct = 0
    linear_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=tqdm_silent)
    for batch_idx, (views, target) in enumerate(pbar):
        view1, view2 = views[0].to(device, non_blocking=True), views[1].to(
            device, non_blocking=True
        )
        target = target.to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp, dtype=dtype):
            features, z1 = model(view1)
            _, z2 = model(view2)
            loss_dict = loss_fn(z1, z2)
            loss = loss_dict["loss"]

        with torch.no_grad():
            features_frozen = features.detach().float()

        linear_outputs = linear_probe(features_frozen)
        linear_loss = F.cross_entropy(linear_outputs, target)

        _, predicted = linear_outputs.max(1)
        linear_correct_batch = predicted.eq(target).sum().item()

        total_loss_batch = loss + linear_loss

        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics dynamically based on loss_dict keys
        for key, value in loss_dict.items():
            if key not in loss_totals:
                loss_totals[key] = 0
            loss_totals[key] += value.item()
        total_linear_loss += linear_loss.item()

        # Update linear probe accuracy (pre-computed under autocast)
        linear_total += target.size(0)
        linear_correct += linear_correct_batch

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Linear": f"{linear_loss.item():.4f}",
                "Acc": f"{100.*linear_correct/linear_total:.2f}%",
            }
        )

    # Update learning rate
    scheduler.step(epoch)

    # Build return dict dynamically
    num_batches = len(train_loader)
    metrics = {key: total / num_batches for key, total in loss_totals.items()}
    metrics["linear_loss"] = total_linear_loss / num_batches
    metrics["linear_acc"] = 100.0 * linear_correct / linear_total

    return metrics


def run(
    fname: str = "geno_jepa/cfgs/genomic.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train a Genomic JEPA (VICReg/BCS) model on genomic data.

    Args:
        fname: Path to YAML config file
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., optim.epochs=50)
    """
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Setup using shared utilities
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    # Create experiment directory using unified structure (if not provided)
    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir = Path(cfg.meta.model_folder)
            folder_name = exp_dir.name
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("image_jepa", cfg)
            exp_dir = get_unified_experiment_dir(
                example_name="image_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        # Extract exp_name from folder name by removing _seed{seed} suffix
        folder_name = exp_dir.name  # e.g., "resnet_vicreg_seed1"
        exp_name = folder_name.rsplit("_seed", 1)[0]  # e.g., "resnet_vicreg"

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "image_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["image_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    # Check if using genomic data or CIFAR-10
    use_genomic = cfg.data.get("use_genomic", False)
    
    if use_genomic:
        logger.info("Loading genomic dataset...")
        
        # Default paths (can be overridden in config)
        base_path = cfg.data.get(
            "base_path",
            "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/chromosome_coordinate"
        )
        
        methylation_path = os.path.join(base_path, "methylation_tensor_chrom_ordered.pkl")
        gene_expression_path = os.path.join(base_path, "gene_expression_tensor_chrom_ordered.pkl")
        labels_path = os.path.join(base_path, "cancer_tags_tensor_chrom_ordered.pkl")
        
        # Get genomic-specific settings
        use_channels = cfg.data.get("use_channels", "both")
        patch_size = cfg.model.get("patch_size") if cfg.model.type == "vit" else None
        
        # Create full dataset (raw data)
        full_dataset_raw = GenomicDataset(
            methylation_path=methylation_path,
            gene_expression_path=gene_expression_path,
            labels_path=labels_path,
            transform=None,
            patch_size=patch_size,
            use_channels=use_channels,
        )
        
        # Split into train/val (80/20 split)
        train_size = int(0.8 * len(full_dataset_raw))
        val_size = len(full_dataset_raw) - train_size
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_dataset_raw)), [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.meta.seed)
        )
        
        # Create training subset with views
        train_sampler = torch.utils.data.Subset(full_dataset_raw, train_indices)
        train_dataset = ImageDataset(
            train_sampler, 
            transform=get_genomic_train_transforms(), 
            num_crops=2
        )
        
        # For validation, use single vector without augmentations
        val_sampler = torch.utils.data.Subset(full_dataset_raw, val_indices)
        val_dataset = ImageDataset(
            val_sampler,
            transform=get_genomic_val_transforms(),
            num_crops=2 # We still use num_crops=2 to keep train_epoch happy, 
                        # or update train_epoch to handle single view
        )
    else:
        logger.info("Loading CIFAR-10 dataset...")
        transform = get_train_transforms()

        # Use EBJEPA_DSETS environment variable if set, otherwise fall back to config
        data_dir = os.environ.get("EBJEPA_DSETS")
        logger.info(f"Using data directory: {data_dir}")

        base_train_dataset = CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )

        train_dataset = ImageDataset(base_train_dataset, transform, num_crops=2)

        val_dataset = CIFAR10(
            root=data_dir, train=False, download=True, transform=get_val_transforms()
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid small batches that cause BatchNorm issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    dataset_name = "Genomic" if use_genomic else "CIFAR-10"
    log_data_info(
        dataset_name,
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )

    # Initialize model
    logger.info("Initializing model...")
    if use_genomic:
        # Determine in_channels from config
        use_channels = cfg.data.get("use_channels", "both")
        in_channels = 2 if use_channels == "both" else 1
        
        if cfg.model.type == "vit":
            # Use 1D Vision Transformer for genomic data
            patch_size = cfg.model.get("patch_size", 100)
            hidden_dim = cfg.model.get("hidden_dim", 256)
            num_layers = cfg.model.get("num_layers", 6)
            num_heads = cfg.model.get("num_heads", 8)
            mlp_dim = cfg.model.get("mlp_dim", 512)
            
            backbone = ViT1DEncoder(
                in_channels=in_channels,
                seq_length=15703,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_dim=mlp_dim
            )
        elif cfg.model.type == "mlp":
            # Use simple MLP encoder for genomic data
            hidden_dim = cfg.model.get("hidden_dim", 1024)
            dropout = cfg.model.get("dropout", 0.2)
            
            backbone = MLPEncoder(
                in_channels=in_channels,
                seq_length=15703,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            # Use 1D Conv encoder for genomic data (default)
            backbone = Conv1DEncoder(in_channels=in_channels)
        features_dim = backbone.features_dim
    elif cfg.model.type == "resnet":
        in_channels = 3
        backbone = ResNet18(in_channels=in_channels)
        features_dim = backbone.features_dim
    elif cfg.model.type == "vit_s":
        features_dim = 384
        image_size = 32
        patch_size = cfg.model.get("patch_size", 8)
        model_kwargs = dict(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=features_dim,
            num_layers=12,
            num_heads=6,
            mlp_dim=4 * features_dim,
        )
        backbone = VisionTransformer(**model_kwargs)
        backbone.heads = nn.Identity()
    elif cfg.model.type == "vit_b":
        features_dim = 768
        image_size = 32
        patch_size = cfg.model.get("patch_size", 8)
        model_kwargs = dict(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=features_dim,
            num_layers=12,
            num_heads=12,
            mlp_dim=4 * features_dim,
        )
        backbone = VisionTransformer(**model_kwargs)
        backbone.heads = nn.Identity()

    model = GenomicSSL(
        backbone,
        features_dim=features_dim,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        proj_output_dim=cfg.model.proj_output_dim,
    )

    if not cfg.model.use_projector:
        model.projector = nn.Identity()

    model = model.to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in backbone.parameters())
    projector_params = (
        sum(p.numel() for p in model.projector.parameters())
        if cfg.model.use_projector
        else 0
    )
    log_model_info(model, {"encoder": encoder_params, "projector": projector_params})

    # Log configuration
    log_config(cfg)

    # Initialize linear probe
    num_classes = 10  # Default for CIFAR-10
    if use_genomic:
        # Get number of classes from the raw genomic dataset
        num_classes = len(torch.unique(full_dataset_raw.labels))
    linear_probe = LinearProbe(feature_dim=features_dim, num_classes=num_classes).to(device)

    # Mixed precision setup
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    use_amp = cfg.training.get("use_amp", True)
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"Using AMP with {dtype=}" if use_amp else f"AMP disabled")

    optimizer = LARS(
        [
            {"params": model.parameters(), "lr": cfg.optim.lr},
            {"params": linear_probe.parameters(), "lr": 0.1},  # Linear probe parameters
        ],
        weight_decay=cfg.optim.weight_decay,
        eta=0.02,
        clip_lr=True,
        exclude_bias_n_norm=True,
        momentum=0.9,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.optim.warmup_epochs,
        max_epochs=cfg.optim.epochs,
        base_lr=cfg.optim.lr,
        min_lr=cfg.optim.min_lr,
        warmup_start_lr=cfg.optim.warmup_start_lr,
    )

    # Initialize loss function
    if cfg.loss.type == "vicreg":
        loss_fn = VICRegLoss(std_coeff=cfg.loss.std_coeff, cov_coeff=cfg.loss.cov_coeff)
    elif cfg.loss.type == "bcs":
        loss_fn = BCS(lmbd=cfg.loss.lmbd)

    # Load checkpoint if requested
    start_epoch = 0
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(ckpt_path, model, optimizer, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        if "linear_probe_state_dict" in ckpt_info:
            linear_probe.load_state_dict(ckpt_info["linear_probe_state_dict"])

    # Training loop
    logger.info(f"Starting training for {cfg.optim.epochs} epochs...")
    start_time = time.time()
    use_amp = cfg.training.get("use_amp", True)
    tqdm_silent = cfg.logging.get("tqdm_silent", False)

    for epoch in range(start_epoch, cfg.optim.epochs):
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            linear_probe,
            scaler,
            device,
            epoch,
            loss_fn,
            use_amp,
            dtype,
            tqdm_silent,
        )

        # Evaluate linear probe on validation set
        val_acc, val_loss = evaluate_linear_probe(
            model, linear_probe, val_loader, device, use_amp
        )

        # Log metrics - dynamically add train_ prefix to all train_metrics keys
        log_dict = {"epoch": epoch}
        for key, value in train_metrics.items():
            log_dict[f"train_{key}"] = value
        log_dict["val_loss"] = val_loss
        log_dict["val_acc"] = val_acc
        log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]

        if wandb_run:
            wandb.log(log_dict)

        # Log progress
        if epoch % cfg.logging.log_every == 0:
            elapsed = time.time() - start_time
            log_epoch(
                epoch,
                {
                    "loss": train_metrics["loss"],
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                total_epochs=cfg.optim.epochs,
                elapsed_time=elapsed,
            )

        # Save checkpoint
        save_checkpoint(
            exp_dir / "latest.pth.tar",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            linear_probe_state_dict=linear_probe.state_dict(),
            linear_val_acc=val_acc,
        )
        if epoch % cfg.logging.save_every == 0 and epoch > 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch}.pth.tar",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                scaler=scaler,
                linear_probe_state_dict=linear_probe.state_dict(),
                linear_val_acc=val_acc,
            )

    logger.info("Training completed!")
    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(run)

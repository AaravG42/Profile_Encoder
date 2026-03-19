"""
Model definitions for Genomic JEPA training.
Contains encoder architectures and loss functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


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

    def __init__(self, in_channels=2, latent_dim=4096):
        super().__init__()
        self.encoder = nn.Sequential(
            # Layer 1: Rapid downsampling (Stride 4)
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32), nn.LeakyReLU(),
            
            # Layer 2: (Stride 4)
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            
            # Layer 3: (Stride 2)
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(),

            # Global Average Pooling collapses the sequence length to 1
            nn.AdaptiveAvgPool1d(1)
        )
        self.features_dim = 128
        # Calculate output size: 17819 -> 8910 -> 4455 -> 2228 -> 1114 -> 557
        # Output: (batch_size, 64, 557)
        # self.flatten_size = 64 * 557  # Flattened feature dimension
        # self.fc = nn.Linear(self.flatten_size, latent_dim)
        # self.features_dim = latent_dim
        # self.features_dim = 64 * 557

    def forward(self, x):
        assert x.dim() == 3, f"Expected input shape (batch_size, C, L), got {x.shape}"
        x = self.encoder(x)
        # x shape: (batch_size, 64, 557)
        x = x.flatten(start_dim=1)  # Flatten to (batch_size, 64*557)
        # x = self.fc(x)              # Compress to latent_dim
        return x


class MLPEncoder(nn.Module):
    """Simple MLP Encoder for genomic data."""

    def __init__(self, in_channels=2, seq_length=17819, hidden_dim=1024, dropout=0.2):
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
        seq_length=17819,
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


class ScGPTEncoder(nn.Module):
    """scGPT-inspired encoder for genomic vectors."""

    def __init__(
        self,
        in_channels=2,
        seq_length=17819,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1,
        num_bins=51,
        max_tokens=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_bins = num_bins
        self.max_tokens = max_tokens

        # Index 0 is reserved for padding. Gene ids start at 1.
        self.gene_embed = nn.Embedding(seq_length + 1, hidden_dim, padding_idx=0)
        self.value_mlps = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(in_channels)]
        )
        self.channel_scale = nn.Parameter(torch.ones(in_channels, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.embed_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.features_dim = hidden_dim

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _bin_channel_values(self, values):
        """
        Bin non-zero values into per-sample quantile bins in [1, num_bins].
        Zero values remain 0.
        """
        batch_size, seq_length = values.shape
        bins = torch.zeros_like(values, dtype=torch.long)

        for sample_idx in range(batch_size):
            sample = values[sample_idx]
            nonzero_mask = sample > 0
            nonzero_idx = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
            nonzero_count = nonzero_idx.numel()

            if nonzero_count == 0:
                continue

            nonzero_values = sample[nonzero_idx]
            sorted_order = torch.argsort(nonzero_values, stable=True)
            ranked_bins = torch.ceil(
                torch.arange(
                    1,
                    nonzero_count + 1,
                    device=values.device,
                    dtype=torch.float32,
                )
                * self.num_bins
                / nonzero_count
            ).to(torch.long)
            ranked_bins = ranked_bins.clamp_(1, self.num_bins)

            assigned_bins = torch.zeros_like(nonzero_idx, dtype=torch.long)
            assigned_bins[sorted_order] = ranked_bins
            bins[sample_idx, nonzero_idx] = assigned_bins

        return bins

    def _select_gene_tokens(self, value_bins):
        """Select the most informative genes and pad to a fixed token length."""
        batch_size, _, seq_length = value_bins.shape
        device = value_bins.device

        gene_ids = torch.zeros(
            batch_size, self.max_tokens, dtype=torch.long, device=device
        )
        selected_bins = torch.zeros(
            batch_size,
            self.in_channels,
            self.max_tokens,
            dtype=torch.long,
            device=device,
        )
        padding_mask = torch.ones(
            batch_size, self.max_tokens, dtype=torch.bool, device=device
        )

        combined_importance = value_bins.sum(dim=1)
        all_gene_ids = torch.arange(1, seq_length + 1, device=device, dtype=torch.long)

        for sample_idx in range(batch_size):
            importance = combined_importance[sample_idx]
            nonzero_idx = (importance > 0).nonzero(as_tuple=False).squeeze(-1)

            if nonzero_idx.numel() == 0:
                keep_idx = torch.tensor([0], device=device, dtype=torch.long)
            else:
                keep_count = min(self.max_tokens, nonzero_idx.numel())
                keep_scores = importance[nonzero_idx]
                topk_order = torch.topk(
                    keep_scores, k=keep_count, largest=True, sorted=True
                ).indices
                keep_idx = nonzero_idx[topk_order]

            token_count = keep_idx.numel()
            gene_ids[sample_idx, :token_count] = all_gene_ids[keep_idx]
            selected_bins[sample_idx, :, :token_count] = value_bins[
                sample_idx, :, keep_idx
            ]
            padding_mask[sample_idx, :token_count] = False

        return gene_ids, selected_bins, padding_mask

    def forward(self, x):
        assert x.dim() == 3, f"Expected input shape (batch_size, C, L), got {x.shape}"
        assert (
            x.shape[1] == self.in_channels
        ), f"Expected {self.in_channels} channels, got {x.shape[1]}"
        assert (
            x.shape[2] == self.seq_length
        ), f"Expected sequence length {self.seq_length}, got {x.shape[2]}"

        channel_bins = [self._bin_channel_values(x[:, idx, :]) for idx in range(self.in_channels)]
        value_bins = torch.stack(channel_bins, dim=1)
        gene_ids, selected_bins, padding_mask = self._select_gene_tokens(value_bins)

        tokens = self.gene_embed(gene_ids)
        for channel_idx, value_mlp in enumerate(self.value_mlps):
            channel_values = (
                selected_bins[:, channel_idx].to(tokens.dtype).unsqueeze(-1)
                / max(self.num_bins, 1)
            )
            tokens = tokens + value_mlp(channel_values) * self.channel_scale[channel_idx]

        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        cls_padding = torch.zeros(
            padding_mask.shape[0], 1, dtype=torch.bool, device=padding_mask.device
        )
        src_key_padding_mask = torch.cat((cls_padding, padding_mask), dim=1)

        x = self.embed_dropout(x)
        x = self.blocks(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x[:, 0]


class GenomicSSL(nn.Module):
    """Genomic Self-Supervised Learning model implementation."""

    def __init__(
        self, 
        backbone, 
        features_dim, 
        proj_hidden_dim=2048, 
        proj_output_dim=2048,
        predictor=None
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
        
        # Predictor (optional)
        self.predictor = predictor

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


class Predictor(nn.Module):
    """Predictor network for JEPA-style training."""

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


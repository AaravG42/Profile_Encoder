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



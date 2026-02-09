"""
I-JEPA (Joint-Embedding Predictive Architecture) adapted for 1D genomic sequences.
Predicts representations of masked genomic regions using visible context.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class GenomicIJEPA(nn.Module):
    """
    I-JEPA for multi-omic genomic data (DNA methylation + gene expression).
    
    Architecture:
    - Context Encoder: Processes visible patches (trainable)
    - Predictor: Predicts target representations from context (trainable)
    - Target Encoder: Generates ground truth representations (EMA-updated)
    """
    
    def __init__(
        self,
        input_dim: int = 15703,
        patch_size: int = 128,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Compute number of patches
        self.n_patches = (input_dim + patch_size - 1) // patch_size
        self.pad_len = self.n_patches * patch_size - input_dim
        
        # Context Encoder (trainable)
        self.context_encoder = GenomicEncoder(
            input_dim=input_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            dropout=dropout
        )
        
        # Target Encoder (EMA-updated, frozen)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Predictor
        self.predictor = PredictorTransformer(
            context_embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            dropout=dropout
        )
        
        # Optional classifier head (for supervised fine-tuning)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def patchify(self, x_dna, x_gene):
        """Convert (B, L) sequences to (B, n_patches, 2*patch_size) patches."""
        B = x_dna.shape[0]
        x = torch.stack([x_dna, x_gene], dim=1)  # (B, 2, L)
        
        if self.pad_len > 0:
            pad = x.new_zeros((B, 2, self.pad_len))
            x = torch.cat([x, pad], dim=2)
        
        # (B, 2, n_patches, patch_size)
        x = x.view(B, 2, self.n_patches, self.patch_size)
        # (B, n_patches, 2, patch_size) -> (B, n_patches, 2*patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, 2 * self.patch_size)
        return x
    
    def forward_target(self, x_dna, x_gene):
        """Target encoder forward (no gradients)."""
        with torch.no_grad():
            patches = self.patchify(x_dna, x_gene)
            h = self.target_encoder(patches)
            h = F.layer_norm(h, (h.size(-1),))
            return h
    
    def forward_context(self, x_dna, x_gene, context_mask=None):
        """Context encoder forward with optional masking."""
        patches = self.patchify(x_dna, x_gene)
        if context_mask is not None:
            # Apply mask: keep only visible patches
            patches = self._apply_mask(patches, context_mask)
        z = self.context_encoder(patches, mask=context_mask)
        return z
    
    def forward_predictor(self, context_features, context_mask, target_mask):
        """Predictor: predict target representations from context."""
        predictions = self.predictor(
            context_features, 
            context_mask, 
            target_mask
        )
        return predictions
    
    def _apply_mask(self, patches, mask):
        """Apply binary mask to patches. mask: (B, n_patches)."""
        B, N, D = patches.shape
        # Create expanded mask
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)
        return patches * mask_expanded
    
    def forward_classifier(self, x_dna, x_gene):
        """Forward pass for classification (using mean-pooled context features)."""
        z = self.forward_context(x_dna, x_gene, context_mask=None)
        z_pooled = z.mean(dim=1)
        logits = self.classifier(z_pooled)
        return logits
    
    def update_target_encoder(self, momentum):
        """EMA update of target encoder from context encoder."""
        with torch.no_grad():
            for param_ctx, param_tgt in zip(
                self.context_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                param_tgt.data.mul_(momentum).add_((1 - momentum) * param_ctx.data)


class GenomicEncoder(nn.Module):
    """Transformer encoder for genomic patches."""
    
    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_patches = (input_dim + patch_size - 1) // patch_size
        
        # Patch embedding (2 channels * patch_size -> embed_dim)
        self.patch_embed = nn.Linear(2 * patch_size, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, patches, mask=None):
        """
        patches: (B, n_patches, 2*patch_size)
        mask: optional (B, n_patches) binary mask (1=keep, 0=ignore)
        """
        B, N, _ = patches.shape
        
        # Embed patches
        x = self.patch_embed(patches)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        x = self.norm(x)
        return x


class PredictorTransformer(nn.Module):
    """
    Predictor: predicts target patch representations from context.
    Takes context features and mask tokens, outputs predictions for targets.
    """
    
    def __init__(
        self,
        context_embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.context_embed_dim = context_embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        
        # Project context features to predictor dimension
        self.context_proj = nn.Linear(context_embed_dim, predictor_embed_dim)
        
        # Learnable mask tokens for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Predictor transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(predictor_embed_dim)
        
        # Project predictions back to context dimension
        self.pred_proj = nn.Linear(predictor_embed_dim, context_embed_dim)
    
    def forward(self, context_features, context_mask, target_mask):
        """
        context_features: (B, n_visible, context_embed_dim)
        context_mask: (B, n_patches) - binary mask for visible patches
        target_mask: (B, n_patches) - binary mask for target patches
        Returns: (B, n_targets, context_embed_dim)
        """
        B = context_features.shape[0]
        
        # Project context features
        x = self.context_proj(context_features)
        
        # Get target positions
        n_targets = int(target_mask.sum(dim=1).max().item())
        
        # Create mask tokens for target positions
        mask_tokens = self.mask_token.expand(B, n_targets, -1)
        
        # Combine context and mask tokens
        # In practice, we process them together with cross-attention
        # For simplicity, concatenate and let transformer handle it
        combined = torch.cat([x, mask_tokens], dim=1)
        
        # Apply transformer blocks
        for block in self.blocks:
            combined = block(combined)
        
        combined = self.norm(combined)
        
        # Extract predictions for target positions (last n_targets tokens)
        predictions = combined[:, -n_targets:, :]
        predictions = self.pred_proj(predictions)
        
        return predictions


class TransformerBlock(nn.Module):
    """Standard Transformer block with self-attention and MLP."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=None if mask is None else (mask == 0)
        )
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class MaskGenerator:
    """
    Generates multi-block masks for I-JEPA training on 1D sequences.
    Creates both context masks (what encoder sees) and target masks (what to predict).
    """
    
    def __init__(
        self,
        n_patches: int,
        num_context_blocks: int = 1,
        num_target_blocks: int = 4,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        target_scale: Tuple[float, float] = (0.15, 0.2),
        min_keep: int = 10,
        allow_overlap: bool = False
    ):
        self.n_patches = n_patches
        self.num_context_blocks = num_context_blocks
        self.num_target_blocks = num_target_blocks
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
    
    def generate_masks(self, batch_size: int, device: torch.device):
        """
        Generate context and target masks for a batch.
        Returns:
            context_mask: (B, n_patches) - 1 for visible, 0 for masked
            target_mask: (B, n_patches) - 1 for prediction targets, 0 otherwise
        """
        context_masks = []
        target_masks = []
        
        for _ in range(batch_size):
            context_mask = self._generate_context_mask()
            target_mask = self._generate_target_mask(context_mask)
            context_masks.append(context_mask)
            target_masks.append(target_mask)
        
        context_masks = torch.stack(context_masks).to(device)
        target_masks = torch.stack(target_masks).to(device)
        
        return context_masks, target_masks
    
    def _generate_context_mask(self):
        """Generate context mask (large contiguous blocks to keep)."""
        mask = torch.zeros(self.n_patches)
        
        for _ in range(self.num_context_blocks):
            # Random block size
            block_size = int(
                self.n_patches * (
                    self.context_scale[0] + 
                    torch.rand(1).item() * (self.context_scale[1] - self.context_scale[0])
                )
            )
            block_size = max(block_size, self.min_keep)
            
            # Random starting position
            start = torch.randint(0, max(1, self.n_patches - block_size + 1), (1,)).item()
            end = min(start + block_size, self.n_patches)
            
            mask[start:end] = 1
        
        # Ensure minimum patches kept
        if mask.sum() < self.min_keep:
            remaining = self.min_keep - int(mask.sum())
            available = torch.where(mask == 0)[0]
            if len(available) > 0:
                indices = available[torch.randperm(len(available))[:remaining]]
                mask[indices] = 1
        
        return mask
    
    def _generate_target_mask(self, context_mask):
        """Generate target mask (blocks to predict)."""
        target_mask = torch.zeros(self.n_patches)
        
        for _ in range(self.num_target_blocks):
            # Random block size
            block_size = int(
                self.n_patches * (
                    self.target_scale[0] + 
                    torch.rand(1).item() * (self.target_scale[1] - self.target_scale[0])
                )
            )
            block_size = max(block_size, 1)
            
            # Random starting position
            max_start = max(1, self.n_patches - block_size + 1)
            start = torch.randint(0, max_start, (1,)).item()
            end = min(start + block_size, self.n_patches)
            
            # Check overlap with context if not allowed
            if not self.allow_overlap:
                if context_mask[start:end].sum() > 0:
                    continue  # Skip this block if overlaps
            
            target_mask[start:end] = 1
        
        # Ensure at least one target patch
        if target_mask.sum() == 0:
            available = torch.arange(self.n_patches)
            if not self.allow_overlap:
                available = torch.where(context_mask == 0)[0]
            if len(available) > 0:
                idx = available[torch.randint(0, len(available), (1,)).item()]
                target_mask[idx] = 1
        
        return target_mask


def momentum_schedule(base_momentum: float, final_momentum: float, epochs: int):
    """Generate momentum schedule for EMA updates."""
    return [
        final_momentum - (final_momentum - base_momentum) * 
        (math.cos(math.pi * i / epochs) + 1) / 2
        for i in range(epochs)
    ]

"""
Compare latent space across multiple training epochs.

Usage:
    python -m examples.image_jepa.compare_epochs \
        --checkpoint_dir /path/to/checkpoint/directory
"""

import os
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from umap import UMAP

from examples.image_jepa.visualize_latent_space import (
    extract_latent_representations,
    get_cancer_type_names,
    load_genomic_dataset,
    load_model_from_checkpoint,
)


def compare_epochs(
    checkpoint_dir: str,
    data_path: str = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/chromosome_coordinate",
    epochs: list = None,
    device: str = "cuda",
    batch_size: int = 32,
):
    """
    Compare latent space visualizations across multiple training epochs.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        data_path: Path to genomic data directory
        epochs: List of epoch numbers to visualize (e.g., [25, 50, 75, 100])
               If None, visualizes all available checkpoints
        device: Device to use (cuda/cpu)
        batch_size: Batch size for feature extraction
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all checkpoint files
    if epochs is None:
        checkpoint_files = sorted(checkpoint_dir.glob("epoch_*.pth.tar"))
        if not checkpoint_files:
            checkpoint_files = [checkpoint_dir / "latest.pth.tar"]
    else:
        checkpoint_files = [checkpoint_dir / f"epoch_{epoch}.pth.tar" for epoch in epochs]
    
    # Filter existing files
    checkpoint_files = [f for f in checkpoint_files if f.exists()]
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint(s) to compare")
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Load dataset once
    print("Loading dataset...")
    dataset = load_genomic_dataset(data_path)
    
    # Get cancer type names
    cancer_mapping = get_cancer_type_names()
    
    # Extract features for each checkpoint
    all_embeddings = []
    epoch_numbers = []
    
    for ckpt_file in checkpoint_files:
        epoch_num = ckpt_file.stem.replace("epoch_", "")
        print(f"\n{'='*80}")
        print(f"Processing {ckpt_file.name} (Epoch {epoch_num})")
        print(f"{'='*80}")
        
        # Load model
        model = load_model_from_checkpoint(str(ckpt_file), device=device)
        
        # Extract features
        features, labels = extract_latent_representations(
            model, dataset, device=device, batch_size=batch_size
        )
        
        # Apply UMAP
        print("Computing UMAP projection...")
        reducer = UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="euclidean",
            random_state=42,
        )
        embedding = reducer.fit_transform(features)
        
        all_embeddings.append(embedding)
        epoch_numbers.append(epoch_num)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Create comparison plot
    print(f"\n{'='*80}")
    print("Creating comparison visualization...")
    print(f"{'='*80}")
    
    n_plots = len(all_embeddings)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes
    
    # Get consistent colors across all plots
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20" if len(unique_labels) <= 20 else "tab20b")
    
    for idx, (embedding, epoch_num, ax) in enumerate(zip(all_embeddings, epoch_numbers, axes)):
        # Plot each cancer type
        for label_idx, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            cancer_name = cancer_mapping.get(label, f"Class {label}")
            
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[cmap(label_idx / len(unique_labels))],
                label=cancer_name,
                alpha=0.6,
                s=15,
                edgecolors="none",
            )
        
        ax.set_xlabel("UMAP Dimension 1", fontsize=10)
        ax.set_ylabel("UMAP Dimension 2", fontsize=10)
        ax.set_title(f"Epoch {epoch_num}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
    
    # Remove empty subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])
    
    # Add shared legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    ncol_legend = 2 if len(unique_labels) > 10 else 1
    fig.legend(
        handles,
        labels_legend,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=ncol_legend,
        fontsize=9,
        markerscale=1.5,
        framealpha=0.9,
    )
    
    plt.suptitle(
        f"Latent Space Evolution - {checkpoint_dir.name}",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    
    plt.tight_layout()
    
    # Save figure
    save_path = checkpoint_dir / "latent_space_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nComparison visualization saved to: {save_path}")
    
    # Save high-res version
    high_res_path = checkpoint_dir / "latent_space_comparison_highres.png"
    plt.savefig(high_res_path, dpi=600, bbox_inches="tight")
    print(f"High-res version saved to: {high_res_path}")
    
    plt.close()
    
    print(f"\n{'='*80}")
    print("Comparison complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    fire.Fire(compare_epochs)

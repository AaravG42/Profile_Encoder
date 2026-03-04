"""
Visualize the latent space of a trained Genomic JEPA model using 3D UMAP with Plotly.

Usage:
    python -m geno_jepa.visualize_latent_space_3d \
        --checkpoint_path /path/to/checkpoint.pth.tar \
        --model_type conv1d \
        --use_channels both \
"""

import os
import pickle
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import VisionTransformer
from tqdm import tqdm
from umap import UMAP
import plotly.graph_objects as go
import plotly.express as px

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
    # Use the new preprocessed data files
    methylation_path = os.path.join(base_path, "Final_Preprocessed_DNA_Methylation_UCSC_PCA_CancerTags.pkl")
    gene_expression_path = os.path.join(base_path, "Final_Preprocessed_Gene_Expression_TCGA_CancerTags.pkl")
    
    dataset = GenomicDataset(
        methylation_path=methylation_path,
        gene_expression_path=gene_expression_path,
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





def plot_umap_3d(features, labels, label_names, save_path="latent_space_umap_3d.html", title="3D Latent Space UMAP"):
    """Create interactive 3D UMAP visualization using Plotly.
    
    Args:
        features: Feature embeddings from the model
        labels: Integer label indices
        label_names: List of cancer type names corresponding to label indices
        save_path: Path to save the visualization
        title: Title for the plot
    """
    print("Computing 3D UMAP projection...")
    print(f"Number of classes: {len(label_names)}")
    
    # Apply UMAP with 3 components
    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=3,  # 3D projection
        metric="euclidean",
        random_state=42,
    )
    
    embedding = reducer.fit_transform(features)
    
    print(f"UMAP embedding shape: {embedding.shape}")
    
    # Create labels array with cancer names
    cancer_names = [label_names[label] if label < len(label_names) else f"Class {label}" for label in labels]
    
    # Create a color mapping for each unique label
    unique_labels = sorted(np.unique(labels))
    color_scale = px.colors.qualitative.Light24 if len(unique_labels) <= 24 else px.colors.qualitative.Alphabet
    color_map = {label: color_scale[i % len(color_scale)] for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]
    
    # Create interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Cancer Type<br>Label",
                thickness=15,
                len=0.7,
            ),
            opacity=0.8,
            line=dict(width=0),
        ),
        text=cancer_names,
        hovertemplate='<b>%{text}</b><br>' +
                      'X: %{x:.2f}<br>' +
                      'Y: %{y:.2f}<br>' +
                      'Z: %{z:.2f}<br>' +
                      '<extra></extra>',
    )])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, family="Arial Black"),
        ),
        scene=dict(
            xaxis=dict(title='UMAP 1', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(title='UMAP 2', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(title='UMAP 3', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        ),
        width=1200,
        height=900,
        hovermode='closest',
        template='plotly_white',
    )
    
    # Save the interactive plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"Interactive 3D UMAP visualization saved to: {save_path}")
    
    # Open in browser
    print(f"\nOpening visualization in browser...")
    fig.show()
    
    # Also create a version with cancer type colors (one trace per cancer type)
    plot_umap_3d_by_cancer(embedding, labels, label_names, 
                           save_path.parent / f"{save_path.stem}_by_cancer.html", 
                           title)


def plot_umap_3d_by_cancer(embedding, labels, label_names, save_path, title):
    """Create 3D UMAP with separate traces for each cancer type for better legend."""
    print("Creating cancer-type colored 3D UMAP...")
    
    fig = go.Figure()
    
    # Get unique labels and create a trace for each
    unique_labels = sorted(np.unique(labels))
    
    # Use a good color palette
    if len(unique_labels) <= 24:
        colors = px.colors.qualitative.Light24
    else:
        colors = px.colors.qualitative.Alphabet
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        cancer_name = label_names[label] if label < len(label_names) else f"Class {label}"
        
        fig.add_trace(go.Scatter3d(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            z=embedding[mask, 2],
            mode='markers',
            name=cancer_name,
            marker=dict(
                size=3,
                color=colors[idx % len(colors)],
                opacity=0.7,
                line=dict(width=0),
            ),
            hovertemplate='<b>' + cancer_name + '</b><br>' +
                          'X: %{x:.2f}<br>' +
                          'Y: %{y:.2f}<br>' +
                          'Z: %{z:.2f}<br>' +
                          '<extra></extra>',
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title + " (By Cancer Type)",
            x=0.5,
            xanchor='center',
            font=dict(size=18, family="Arial Black"),
        ),
        scene=dict(
            xaxis=dict(title='UMAP 1', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(title='UMAP 2', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(title='UMAP 3', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        ),
        width=1400,
        height=900,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1,
        ),
        showlegend=True,
    )
    
    # Save the interactive plot
    fig.write_html(str(save_path))
    print(f"Cancer-type colored 3D UMAP saved to: {save_path}")
    
    # Also show in browser
    fig.show()


def main(
    checkpoint_path: str = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/Aarav_exps/eb_jepa/checkpoints/image_jepa/dev_2026-02-10_05-01/conv1d_vicreg_proj_bs32_ep150_ph2048_po2048_std1.0_cov80.0_seed42/latest.pth.tar",
    data_path: str = "/data/TCGA_cleaned",
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
    Visualize latent space of trained Genomic JEPA model in 3D using Plotly.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to genomic data directory
        save_path: Path to save visualization HTML (default: next to checkpoint)
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
        save_path = checkpoint_dir / f"latent_space_umap_3d_{epoch}.html"
    
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
    
    # Get label names from the dataset
    label_names = dataset.label_names
    print(f"\nDetected {len(label_names)} cancer types:")
    for i, name in enumerate(label_names):
        print(f"  {i}: {name}")
    
    # Save features and labels for future use
    features_save_path = Path(save_path).parent / "latent_features_3d.npz"
    np.savez(features_save_path, features=features, labels=labels, label_names=label_names)
    print(f"Features saved to: {features_save_path}")
    
    # Create 3D UMAP visualization
    checkpoint_name = Path(checkpoint_path).parent.name
    epoch_name = Path(checkpoint_path).stem
    title = f"3D Latent Space UMAP - {checkpoint_name} ({epoch_name})"
    
    plot_umap_3d(features, labels, label_names, save_path=save_path, title=title)
    
    print("\n3D Visualization complete!")
    print(f"Main plot: {save_path}")
    print(f"Features: {features_save_path}")
    print(f"\nThe interactive 3D visualization should open in your browser.")
    print(f"You can rotate, zoom, and hover over points to see details.")


if __name__ == "__main__":
    fire.Fire(main)

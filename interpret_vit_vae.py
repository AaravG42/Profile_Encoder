
import os
import pickle
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Genotype_Induced_Drug_Design.PVAE.Aarav_exps.vit_vae import ViTVAE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpret():
    # Paths
    base_dir = "/home/dmlab/Devendra"
    pvae_dir = os.path.join(base_dir, "Genotype_Induced_Drug_Design/PVAE")
    coord_dir = os.path.join(pvae_dir, "chromosome_coordinate")
    model_path = os.path.join(pvae_dir, "Aarav_exps/vit_vae_supervised_model_coordinate.pt")
    
    # Load mappings
    with open(os.path.join(base_dir, "cancer_class_mapping.json"), "r") as f:
        cancer_mapping = json.load(f)
        
    with open(os.path.join(coord_dir, "gene_index_mapping_chrom_ordered.json"), "r") as f:
        gene_mapping = json.load(f)
        
    # Model parameters from training script
    input_dim = 15703
    patch_size = 128
    z_dim = 128
    embed_dim = 256
    num_layers = 6
    num_heads = 8
    
    # Load labels to determine number of classes and sample some data
    with open(os.path.join(coord_dir, "cancer_tags_tensor_chrom_ordered.pkl"), "rb") as f:
        labels_raw = pickle.load(f)
    
    if labels_raw.dim() > 1 and labels_raw.shape[1] > 1:
        num_classes = labels_raw.shape[1]
        labels = torch.argmax(labels_raw, dim=1)
    else:
        num_classes = len(torch.unique(labels_raw))
        labels = labels_raw
    
    print(f"Detected {num_classes} classes.")
    
    # Initialize and load model
    model = ViTVAE(
        input_dim=input_dim,
        z_dim=z_dim,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    model.to(device)
    model.eval()
    
    # Load feature data
    with open(os.path.join(coord_dir, "methylation_tensor_chrom_ordered.pkl"), "rb") as f:
        dna_meth = pickle.load(f)
    with open(os.path.join(coord_dir, "gene_expression_tensor_chrom_ordered.pkl"), "rb") as f:
        gene_exp = pickle.load(f)
        
    # Re-implement encoder part to capture attention
    @torch.no_grad()
    def get_attention(x_dna, x_gene):
        x_dna = x_dna.to(device)
        x_gene = x_gene.to(device)
        
        patches = model.patchify(x_dna, x_gene)
        emb = model.patch_embed(patches)
        emb = emb + model.pos_embed
        
        all_attens = []
        x = emb
        for layer in model.transformer.layers:
            # MultiheadAttention call inside TransformerEncoderLayer usually has need_weights=False
            # We call it manually to get weights
            _, weights = layer.self_attn(x, x, x, need_weights=True)
            # Use the layer's forward pass to proceed to the next layer
            x = layer(x)
            all_attens.append(weights.cpu())
            
        return all_attens

    # Target cancer types to interpret
    target_cancers = ["BRCA", "LUAD", "MESO", "GBM", "KIRC"]
    # Map cancer names back to indices
    inv_cancer_mapping = {v: k for k, v in cancer_mapping.items()}
    
    print("\nStarting Interpretation of Attention Maps...")
    print("-" * 50)
    
    for cancer_name in target_cancers:
        if cancer_name not in inv_cancer_mapping:
            continue
            
        lab_idx = int(inv_cancer_mapping[cancer_name])
        # Find all samples with this label
        sample_indices = (labels == lab_idx).nonzero(as_tuple=True)[0]
        
        if len(sample_indices) == 0:
            print(f"No samples found for cancer type: {cancer_name}")
            continue
            
        # Take the first sample
        idx = sample_indices[0].item()
        
        x_dna = dna_meth[idx:idx+1]
        x_gene = gene_exp[idx:idx+1]
        
        attns = get_attention(x_dna, x_gene)
        # attns is list of 6 tensors of shape (1, 123, 123)
        
        # Average attention across all layers
        # Or take the last layer? Usually last layer or mean of all.
        avg_attn = torch.stack(attns).mean(dim=0).squeeze(0) # (123, 123)
        
        # Importance score for each patch: mean attention received from all other patches
        importance = avg_attn.mean(dim=0) # (123,)
        
        # Get top 5 patches
        top_k = 5
        top_values, top_indices = torch.topk(importance, top_k)
        
        print(f"\nCancer Type: {cancer_name} (Sample Index: {idx})")
        for i in range(top_k):
            p_idx = top_indices[i].item()
            p_val = top_values[i].item()
            
            # Map back to genes
            start_gene_idx = p_idx * patch_size
            end_gene_idx = min((p_idx + 1) * patch_size, input_dim)
            
            patch_genes = [gene_mapping.get(str(g), "Unknown") for g in range(start_gene_idx, end_gene_idx)]
            # Filter out and format
            gene_str = ", ".join(patch_genes[:8]) + " ..."
            
            print(f"  Rank {i+1}: Patch {p_idx} (Score: {p_val:.4f})")
            print(f"    Genes: {gene_str}")

    print("-" * 50)
    print("Interpretation complete.")

if __name__ == "__main__":
    interpret()

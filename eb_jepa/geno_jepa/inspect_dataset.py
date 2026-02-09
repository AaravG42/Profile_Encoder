"""
Helper script to inspect the genomic dataset and determine cancer type mappings.

Usage:
    python -m examples.image_jepa.inspect_dataset
"""

import os
import pickle

import fire
import numpy as np
import torch


def inspect_dataset(
    data_path: str = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/chromosome_coordinate",
):
    """
    Inspect the genomic dataset to determine class distribution and cancer types.
    
    Args:
        data_path: Path to genomic data directory
    """
    print("=" * 80)
    print("Genomic Dataset Inspector")
    print("=" * 80)
    
    # Load data
    methylation_path = os.path.join(data_path, "methylation_tensor_chrom_ordered.pkl")
    gene_expression_path = os.path.join(data_path, "gene_expression_tensor_chrom_ordered.pkl")
    labels_path = os.path.join(data_path, "cancer_tags_tensor_chrom_ordered.pkl")
    
    print(f"\nData directory: {data_path}")
    print(f"Methylation file: {os.path.basename(methylation_path)}")
    print(f"Gene expression file: {os.path.basename(gene_expression_path)}")
    print(f"Labels file: {os.path.basename(labels_path)}")
    
    # Load methylation
    print("\n" + "-" * 80)
    print("Loading methylation data...")
    with open(methylation_path, "rb") as f:
        methylation = pickle.load(f)
    print(f"  Shape: {methylation.shape}")
    print(f"  Dtype: {methylation.dtype}")
    print(f"  Range: [{methylation.min():.4f}, {methylation.max():.4f}]")
    print(f"  Mean: {methylation.mean():.4f}, Std: {methylation.std():.4f}")
    
    # Load gene expression
    print("\n" + "-" * 80)
    print("Loading gene expression data...")
    with open(gene_expression_path, "rb") as f:
        gene_expression = pickle.load(f)
    print(f"  Shape: {gene_expression.shape}")
    print(f"  Dtype: {gene_expression.dtype}")
    print(f"  Range: [{gene_expression.min():.4f}, {gene_expression.max():.4f}]")
    print(f"  Mean: {gene_expression.mean():.4f}, Std: {gene_expression.std():.4f}")
    
    # Load labels
    print("\n" + "-" * 80)
    print("Loading labels...")
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    
    print(f"  Shape: {labels.shape}")
    print(f"  Dtype: {labels.dtype}")
    
    # Handle one-hot encoded labels
    if labels.dim() > 1 and labels.shape[1] > 1:
        print(f"  Format: One-hot encoded ({labels.shape[1]} classes)")
        labels_indices = torch.argmax(labels, dim=1)
    else:
        print(f"  Format: Class indices")
        labels_indices = labels
    
    # Class distribution
    print("\n" + "-" * 80)
    print("Class Distribution:")
    print("-" * 80)
    
    unique_labels, counts = torch.unique(labels_indices, return_counts=True)
    
    print(f"\nTotal samples: {len(labels_indices)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"\nClass ID | Count | Percentage")
    print("-" * 40)
    
    for label, count in zip(unique_labels, counts):
        percentage = 100 * count / len(labels_indices)
        print(f"{label:8d} | {count:5d} | {percentage:6.2f}%")
    
    # Check if we have cancer type name mapping
    print("\n" + "-" * 80)
    print("Checking for cancer type name mappings...")
    
    # Try to load cancer_class_mapping.json if it exists
    mapping_path = os.path.join(
        os.path.dirname(data_path),
        "cancer_class_mapping.json"
    )
    
    if os.path.exists(mapping_path):
        import json
        with open(mapping_path, "r") as f:
            cancer_mapping = json.load(f)
        
        print(f"\nFound cancer type mapping at: {mapping_path}")
        print("\nClass ID | Cancer Type | Count")
        print("-" * 60)
        
        for label, count in zip(unique_labels, counts):
            label_str = str(int(label))
            cancer_name = cancer_mapping.get(label_str, f"Unknown ({label})")
            percentage = 100 * count / len(labels_indices)
            print(f"{label:8d} | {cancer_name:20s} | {count:5d} ({percentage:5.2f}%)")
        
        # Print Python dictionary format for easy copying
        print("\n" + "=" * 80)
        print("Python Dictionary (copy this to visualize_latent_space.py):")
        print("=" * 80)
        print("cancer_mapping = {")
        for label in sorted(unique_labels.tolist()):
            label_str = str(int(label))
            cancer_name = cancer_mapping.get(label_str, f"Class_{label}")
            print(f'    {int(label)}: "{cancer_name}",')
        print("}")
    else:
        print(f"\nNo cancer type mapping found at: {mapping_path}")
        print("Using generic class names (Class 0, Class 1, etc.)")
        
        # Print template
        print("\n" + "=" * 80)
        print("Template for cancer type mapping (update with actual names):")
        print("=" * 80)
        print("cancer_mapping = {")
        for label in sorted(unique_labels.tolist()):
            print(f'    {int(label)}: "CancerType_{int(label)}",  # TODO: Update with actual name')
        print("}")
    
    # Data validation
    print("\n" + "=" * 80)
    print("Data Validation:")
    print("=" * 80)
    
    assert methylation.shape == gene_expression.shape, "Methylation and gene expression shapes don't match!"
    assert methylation.shape[0] == len(labels_indices), "Number of samples doesn't match labels!"
    assert methylation.shape[1] == 15703, f"Expected 15703 features, got {methylation.shape[1]}"
    
    print("✓ All validation checks passed!")
    
    # Reshaping test
    print("\n" + "-" * 80)
    print("Testing 2D reshaping (15703 → 383 x 41):")
    print("-" * 80)
    
    test_sample = methylation[0]
    reshaped = test_sample.reshape(383, 41)
    print(f"Original shape: {test_sample.shape}")
    print(f"Reshaped: {reshaped.shape}")
    print(f"Total elements preserved: {test_sample.numel() == reshaped.numel()}")
    
    print("\n" + "=" * 80)
    print("Inspection complete!")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(inspect_dataset)

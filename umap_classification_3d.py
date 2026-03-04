import os
import pickle
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(data_path):
    """Load the genomic dataset from pickle files."""
    print(f"Loading data from {data_path}...")
    meth_path = os.path.join(data_path, "methylation_tensor_chrom_ordered.pkl")
    gene_path = os.path.join(data_path, "gene_expression_tensor_chrom_ordered.pkl")
    label_path = os.path.join(data_path, "cancer_tags_tensor_chrom_ordered.pkl")
    
    with open(meth_path, "rb") as f:
        methylation = pickle.load(f).float().numpy()
    with open(gene_path, "rb") as f:
        gene_expression = pickle.load(f).float().numpy()
    with open(label_path, "rb") as f:
        labels = pickle.load(f)
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)
        labels = labels.numpy()
        
    # Combine features: (N, 15703) + (N, 15703) -> (N, 31406)
    features = np.concatenate([gene_expression, methylation], axis=1)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    return features, labels

def main():
    # Configuration
    data_path = "/home/aarav/data/chromosome_coordinate"
    cancer_mapping = {
        0: "ACC", 1: "BLCA", 2: "BRCA", 3: "CESC", 4: "COAD", 5: "ESCA", 6: "GBM",
        7: "HNSC", 8: "KICH", 9: "KIRC", 10: "KIRP", 11: "LGG", 12: "LIHC", 13: "LUAD",
        14: "LUSC", 15: "MESO", 16: "OV", 17: "PAAD", 18: "PCPG", 19: "PRAD", 20: "READ",
        21: "SARC", 22: "SKCM", 23: "STAD", 24: "TGCT", 25: "THCA", 26: "UCEC", 27: "UCS"
    }

    # 1. Load Data
    features, labels = load_data(data_path)
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 3. Simple Classification (Random Forest on raw features)
    print(\"\\nTraining Random Forest classifier on raw features...\")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy (Raw Features): {acc:.4f}")

    # 4. UMAP Dimensionality Reduction (3D)
    print(\"\\nComputing 3D UMAP projection...\")
    reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    # Fit UMAP on a subset if dataset is too large, but here we'll try full
    embedding = reducer.fit_transform(features)
    
    # 5. Classification on UMAP features
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        embedding, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf_u = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf_u.fit(X_train_u, y_train_u)
    y_pred_u = clf_u.predict(X_test_u)
    acc_u = accuracy_score(y_test_u, y_pred_u)
    print(f"Classification Accuracy (UMAP Features): {acc_u:.4f}")

    # 6. 3D Visualization
    print(\"\\nGenerating 3D Visualization...\")
    cancer_names = [cancer_mapping.get(label, f"Class {label}") for label in labels]
    
    fig = go.Figure()
    unique_labels = sorted(np.unique(labels))
    
    # Use a large discrete color palette
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = cancer_mapping.get(label, f"Class {label}")
        fig.add_trace(go.Scatter3d(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            z=embedding[mask, 2],
            mode='markers',
            marker=dict(size=3, opacity=0.7, color=colors[i % len(colors)]),
            name=name,
            text=[name] * np.sum(mask),
            hoverinfo='text'
        ))

    fig.update_layout(
        title=\"3D UMAP Visualization of Genomic Data\",
        scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'),
        width=1000, height=800,
        legend=dict(itemsizing='constant')
    )
    
    save_name = \"genomic_umap_3d.html\"
    fig.write_html(save_name)
    print(f\"\\n3D Visualization saved to {save_name}\")
    
    # Show summary
    print(\"\\nSummary Results:\")
    print(f\"- Raw Feature Accuracy:  {acc:.4f}\")
    print(f\"- UMAP Feature Accuracy: {acc_u:.4f}\")

if __name__ == \"__main__\":
    main()

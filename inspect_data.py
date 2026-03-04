import pandas as pd
import pickle

tpm_path = '/data/Final_Preprocessed_Gene_Expression_TCGA_CancerTags.pkl'
meth_path = '/data/Final_Preprocessed_DNA_Methylation_UCSC_PCA_CancerTags.pkl'

print("--- Gene Expression (TPM) ---")
try:
    with open(tpm_path, 'rb') as f:
        tpm_data = pickle.load(f)
    print(f"Shape: {tpm_data.shape}")
    print(tpm_data.iloc[:10, :5])
except Exception as e:
    print(f"Error loading TPM: {e}")

print("\n--- DNA Methylation ---")
try:
    with open(meth_path, 'rb') as f:
        meth_data = pickle.load(f)
    print(f"Shape: {meth_data.shape}")
    print(meth_data.iloc[:10, :5])
except Exception as e:
    print(f"Error loading DNA Methylation: {e}")

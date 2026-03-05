import pandas as pd
import pickle
import numpy as np

# Load the clinical/prognosis data
print("Loading clinical prognosis data...")
try:
    clinical_data = pd.read_csv('/data/Prognosis_TCGA_PANCAN/clinical_PANCAN_patient_with_followup.tsv', sep='\t', encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 encoding failed, trying latin-1...")
    clinical_data = pd.read_csv('/data/Prognosis_TCGA_PANCAN/clinical_PANCAN_patient_with_followup.tsv', sep='\t', encoding='latin-1')
except Exception as e:
    print(f"ISO-8859-1 failed, trying with encoding errors='ignore'...")
    clinical_data = pd.read_csv('/data/Prognosis_TCGA_PANCAN/clinical_PANCAN_patient_with_followup.tsv', sep='\t', encoding='utf-8', encoding_errors='ignore')

print(f"Clinical data shape: {clinical_data.shape}")
print(f"Clinical data columns (first 10): {list(clinical_data.columns[:10])}")

# Check if there's a patient ID column
if 'bcr_patient_barcode' in clinical_data.columns:
    clinical_patients = set(clinical_data['bcr_patient_barcode'].values)
    print(f"\nNumber of patients in clinical data: {len(clinical_patients)}")
    print(f"Sample patient IDs (first 5): {list(clinical_patients)[:5]}")
else:
    print("\nSearching for patient ID column...")
    print(f"Available columns: {list(clinical_data.columns)}")
    # Try to find the patient ID column
    for col in clinical_data.columns:
        if 'patient' in col.lower() or 'barcode' in col.lower():
            print(f"Potential patient ID column: {col}")

# Load the gene expression data
print("\n" + "="*80)
print("Loading gene expression data...")
with open('/data/TCGA_cleaned/Final_Preprocessed_Gene_Expression_TCGA_CancerTags.pkl', 'rb') as f:
    gene_data = pickle.load(f)

print(f"Gene expression data type: {type(gene_data)}")

if isinstance(gene_data, pd.DataFrame):
    print(f"Gene expression data shape: {gene_data.shape}")
    print(f"Columns (patient names) - first 10: {list(gene_data.columns[:10])}")
    print(f"Index (genes?) - first 10: {list(gene_data.index[:10])}")
    
    gene_patients = set(gene_data.columns)
    print(f"\nNumber of patients in gene expression data: {len(gene_patients)}")
    
elif isinstance(gene_data, dict):
    print(f"Gene expression data keys: {gene_data.keys()}")
    # Try to find the actual data
    for key, value in gene_data.items():
        print(f"\nKey: {key}, Type: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")
        if hasattr(value, 'columns'):
            print(f"Columns (first 10): {list(value.columns[:10])}")
            gene_patients = set(value.columns)
            break
else:
    print(f"Unexpected data structure: {type(gene_data)}")
    if hasattr(gene_data, 'shape'):
        print(f"Shape: {gene_data.shape}")

# Compare the patient IDs
print("\n" + "="*80)
print("ALIGNMENT ANALYSIS")
print("="*80)

if 'bcr_patient_barcode' in clinical_data.columns and 'gene_patients' in locals():
    clinical_patients = set(clinical_data['bcr_patient_barcode'].values)
    
    # Find overlapping patients
    overlap = clinical_patients.intersection(gene_patients)
    only_clinical = clinical_patients - gene_patients
    only_gene = gene_patients - clinical_patients
    
    print(f"\nPatients in both datasets: {len(overlap)}")
    print(f"Patients only in clinical data: {len(only_clinical)}")
    print(f"Patients only in gene expression data: {len(only_gene)}")
    
    print(f"\nOverlap percentage (based on clinical): {len(overlap)/len(clinical_patients)*100:.2f}%")
    print(f"Overlap percentage (based on gene expression): {len(overlap)/len(gene_patients)*100:.2f}%")
    
    # Show some examples
    print(f"\nSample overlapping patient IDs (first 5): {list(overlap)[:5]}")
    
    if len(only_clinical) > 0:
        print(f"\nSample patients only in clinical (first 5): {list(only_clinical)[:5]}")
    
    if len(only_gene) > 0:
        print(f"\nSample patients only in gene expression (first 5): {list(only_gene)[:5]}")
    
    # Check if patient ID formats might be different (e.g., truncated)
    print("\n" + "="*80)
    print("CHECKING ID FORMAT DIFFERENCES")
    print("="*80)
    
    # Take first clinical patient and first gene patient
    sample_clinical = list(clinical_patients)[0]
    sample_gene = list(gene_patients)[0]
    
    print(f"\nSample clinical patient ID: '{sample_clinical}' (length: {len(sample_clinical)})")
    print(f"Sample gene expression patient ID: '{sample_gene}' (length: {len(sample_gene)})")
    
    # Check if gene IDs might be truncated versions of clinical IDs
    matching_by_truncation = 0
    for gene_id in list(gene_patients)[:100]:  # Check first 100
        for clinical_id in clinical_patients:
            if gene_id in clinical_id or clinical_id in gene_id:
                matching_by_truncation += 1
                break
    
    print(f"\nOut of first 100 gene IDs, {matching_by_truncation} match (partially) with clinical IDs")
    
    # Try removing the suffix (e.g., '-01') from gene expression IDs
    print("\n" + "="*80)
    print("MATCHING WITH TRUNCATED IDs (removing suffix)")
    print("="*80)
    
    # Remove the last part (e.g., '-01') from gene expression patient IDs
    gene_patients_truncated = set()
    gene_id_mapping = {}  # Maps truncated ID to full ID
    
    for gene_id in gene_patients:
        # Remove the '-01', '-02', etc. suffix (last 3 characters if they match the pattern)
        if len(gene_id) > 12 and gene_id[-3] == '-':
            truncated = gene_id[:-3]
            gene_patients_truncated.add(truncated)
            if truncated not in gene_id_mapping:
                gene_id_mapping[truncated] = []
            gene_id_mapping[truncated].append(gene_id)
        else:
            gene_patients_truncated.add(gene_id)
            gene_id_mapping[gene_id] = [gene_id]
    
    # Find overlapping patients with truncated IDs
    overlap_truncated = clinical_patients.intersection(gene_patients_truncated)
    only_clinical_truncated = clinical_patients - gene_patients_truncated
    only_gene_truncated = gene_patients_truncated - clinical_patients
    
    print(f"\nPatients in both datasets (after truncation): {len(overlap_truncated)}")
    print(f"Patients only in clinical data: {len(only_clinical_truncated)}")
    print(f"Patients only in gene expression data: {len(only_gene_truncated)}")
    
    print(f"\nOverlap percentage (based on clinical): {len(overlap_truncated)/len(clinical_patients)*100:.2f}%")
    print(f"Overlap percentage (based on gene expression): {len(overlap_truncated)/len(gene_patients_truncated)*100:.2f}%")
    
    print(f"\nSample overlapping patient IDs (first 10): {list(overlap_truncated)[:10]}")
    
    # Check if any truncated IDs map to multiple full IDs
    multiple_mappings = {k: v for k, v in gene_id_mapping.items() if len(v) > 1}
    if multiple_mappings:
        print(f"\n{len(multiple_mappings)} patient IDs have multiple samples:")
        for i, (truncated, full_ids) in enumerate(list(multiple_mappings.items())[:5]):
            print(f"  {truncated}: {full_ids}")
            if i >= 4:
                break
    
    # Save the alignment mapping to a file for future use
    print("\n" + "="*80)
    print("SAVING ALIGNMENT DATA")
    print("="*80)
    
    alignment_data = {
        'overlapping_patients': list(overlap_truncated),
        'only_clinical': list(only_clinical_truncated),
        'only_gene_expression': list(only_gene_truncated),
        'gene_id_mapping': gene_id_mapping,
        'stats': {
            'total_clinical': len(clinical_patients),
            'total_gene_expression': len(gene_patients),
            'total_gene_expression_truncated': len(gene_patients_truncated),
            'overlap': len(overlap_truncated),
            'overlap_pct_clinical': len(overlap_truncated)/len(clinical_patients)*100,
            'overlap_pct_gene': len(overlap_truncated)/len(gene_patients_truncated)*100
        }
    }
    
    import json
    with open('/home/aarav/Profile_VAE/patient_alignment_prognosis.json', 'w') as f:
        json.dump(alignment_data, f, indent=2)
    
    print("Alignment results saved to: patient_alignment_prognosis.json")

print("\n" + "="*80)
print("Analysis complete!")

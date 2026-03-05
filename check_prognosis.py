import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/data/Prognosis_TCGA_PANCAN/clinical_PANCAN_patient_with_followup.tsv', 
                 sep='\t', encoding='latin-1', low_memory=False)

print("="*80)
print("PROGNOSIS DATA SUMMARY")
print("="*80)
print(f"Patients: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print("\n" + "="*80)
print("KEY SURVIVAL/OUTCOME COLUMNS")
print("="*80)

# Vital status
if 'vital_status' in df.columns:
    print(f"\nVital Status ({df['vital_status'].notna().sum():,} patients):")
    for status, count in df['vital_status'].value_counts().head(5).items():
        print(f"  {status}: {count:,}")

# Days to death
if 'days_to_death' in df.columns:
    death_data = pd.to_numeric(df['days_to_death'], errors='coerce')
    deceased = death_data.notna().sum()
    if deceased > 0:
        mean_days = death_data.mean()
        median_days = death_data.median()
        print(f"\nDays to Death ({deceased:,} patients died):")
        print(f"  Mean: {mean_days:.0f} days ({mean_days/365:.1f} years)")
        print(f"  Median: {median_days:.0f} days ({median_days/365:.1f} years)")

# Days to followup
if 'days_to_last_followup' in df.columns:
    followup_data = pd.to_numeric(df['days_to_last_followup'], errors='coerce')
    followup = followup_data.notna().sum()
    if followup > 0:
        mean_days = followup_data.mean()
        print(f"\nDays to Last Followup ({followup:,} patients):")
        print(f"  Mean: {mean_days:.0f} days ({mean_days/365:.1f} years)")

print("\n" + "="*80)
print("CANCER TYPES (TOP 15)")
print("="*80)
if 'acronym' in df.columns:
    for cancer, count in df['acronym'].value_counts().head(15).items():
        print(f"  {cancer:6s}: {count:4d}")

print("\n" + "="*80)
print("OTHER KEY CLINICAL COLUMNS")
print("="*80)
other_cols = ['gender', 'age_at_initial_pathologic_diagnosis', 'tumor_status', 
              'pathologic_stage', 'race', 'ethnicity']
for col in other_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,}/{len(df):,} ({non_null/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("SAMPLE OF ALL COLUMNS (first 30)")
print("="*80)
for i, col in enumerate(df.columns[:30], 1):
    print(f"  {i:2d}. {col}")
print(f"  ... and {len(df.columns)-30} more columns")

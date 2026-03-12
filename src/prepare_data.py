import pandas as pd
import numpy as np
import os

print("=" * 60)
print("RARE-AI: Full Data Preparation Pipeline")
print("=" * 60)

data_path = "data/raw/"
out_path  = "data/processed/"
os.makedirs(out_path, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD ALL DATASETS
# ─────────────────────────────────────────────────────────────
print("\n[1/8] Loading all datasets...")

# Orphanet rare disease files
phenotypes   = pd.read_csv(data_path + "orphadata_phenotypes.csv")
complete     = pd.read_csv(data_path + "rare_diseases_complete.csv")
info         = pd.read_csv(data_path + "rare_diseases_info.csv")
nat_history  = pd.read_csv(data_path + "rare_diseases_natural_history.csv")
prevalence   = pd.read_csv(data_path + "rare_diseases_prevalence.csv")
genes        = pd.read_csv(data_path + "rare_diseases_genes.csv")

# Kaggle common disease files
kaggle_symptoms   = pd.read_csv(data_path + "dataset.csv")
kaggle_severity   = pd.read_csv(data_path + "Symptom-severity.csv")
kaggle_profile    = pd.read_csv(data_path + "Disease_symptom_and_patient_profile_dataset.csv")
kaggle_desc       = pd.read_csv(data_path + "symptom_Description.csv")
kaggle_precaution = pd.read_csv(data_path + "symptom_precaution.csv")

# HPOA phenotypes — optional, loaded if present
hpoa_path = data_path + "hpoa_phenotypes.csv"
if os.path.exists(hpoa_path):
    hpoa = pd.read_csv(hpoa_path, low_memory=False)
    print(f"  hpoa_phenotypes:        {len(hpoa):,} rows ✅")
    HAS_HPOA = True
else:
    hpoa = None
    HAS_HPOA = False
    print(f"  hpoa_phenotypes:        NOT FOUND — skipping (place hpoa_phenotypes.csv in data/raw/ to include)")

print(f"  orphadata_phenotypes:   {len(phenotypes):,} rows | {phenotypes['OrphaCode'].nunique():,} rare diseases")
print(f"  rare_diseases_complete: {len(complete):,} rows")
print(f"  rare_diseases_info:     {len(info):,} rows | cols: {info.columns.tolist()}")
print(f"  rare_diseases_genes:    {len(genes):,} rows")
print(f"  natural_history:        {len(nat_history):,} rows")
print(f"  prevalence:             {len(prevalence):,} rows")
print(f"  kaggle dataset:         {len(kaggle_symptoms):,} rows | {kaggle_symptoms['Disease'].nunique()} diseases")
print(f"  kaggle patient profile: {len(kaggle_profile):,} rows | {kaggle_profile['Disease'].nunique()} diseases")
print(f"  symptom severity:       {len(kaggle_severity):,} symptoms")

# ─────────────────────────────────────────────────────────────
# 2. KAGGLE SYMPTOM DATASET → LONG FORMAT
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Processing Kaggle symptom dataset...")

symptom_cols = [c for c in kaggle_symptoms.columns if c.startswith("Symptom")]
kaggle_long  = kaggle_symptoms.melt(
    id_vars=["Disease"], value_vars=symptom_cols, value_name="Symptom"
).dropna(subset=["Symptom"])
kaggle_long["Symptom"]  = kaggle_long["Symptom"].str.strip().str.lower().str.replace(" ", "_")
kaggle_long             = kaggle_long[kaggle_long["Symptom"] != ""].drop(columns=["variable"])
kaggle_long["Disease"]  = kaggle_long["Disease"].str.strip()
kaggle_severity["Symptom"] = kaggle_severity["Symptom"].str.strip().str.lower().str.replace(" ", "_")
kaggle_long             = kaggle_long.merge(kaggle_severity, on="Symptom", how="left")
kaggle_long["weight"]         = kaggle_long["weight"].fillna(3)
kaggle_long["FrequencyScore"] = (kaggle_long["weight"] / kaggle_long["weight"].max()).round(3)
kaggle_long             = kaggle_long.rename(columns={"Disease": "DiseaseName", "Symptom": "HPO_Term"})
kaggle_long["Source"]    = "kaggle"
kaggle_long["OrphaCode"] = None
print(f"  Done: {len(kaggle_long):,} rows | {kaggle_long['DiseaseName'].nunique()} diseases")

# ─────────────────────────────────────────────────────────────
# 3. KAGGLE PATIENT PROFILE → LONG FORMAT
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Processing Kaggle patient profile...")

binary_cols  = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
profile_long = kaggle_profile.melt(
    id_vars=["Disease"], value_vars=binary_cols,
    var_name="HPO_Term", value_name="Present"
)
profile_long = profile_long[profile_long["Present"] == "Yes"].copy()
profile_long["HPO_Term"]       = profile_long["HPO_Term"].str.lower().str.replace(" ", "_")
profile_long["FrequencyScore"] = 0.7
profile_long["Source"]         = "kaggle_profile"
profile_long["OrphaCode"]      = None
profile_long = profile_long.rename(columns={"Disease": "DiseaseName"})
profile_long = profile_long[["OrphaCode","DiseaseName","HPO_Term","FrequencyScore","Source"]]
print(f"  Done: {len(profile_long):,} rows | {profile_long['DiseaseName'].nunique()} diseases")

# ─────────────────────────────────────────────────────────────
# 4. ORPHANET PHENOTYPES → NUMERIC FREQUENCY SCORE
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Processing Orphanet phenotypes...")

frequency_map = {
    "Obligate (100%)":        1.0,
    "Very frequent (99-80%)": 0.9,
    "Frequent (79-30%)":      0.6,
    "Occasional (29-5%)":     0.3,
    "Very rare (<4-1%)":      0.1,
    "Excluded (0%)":          0.0,
}
phenotypes["FrequencyScore"] = phenotypes["Frequency"].map(frequency_map).fillna(0.3)
pheno_filtered               = phenotypes[phenotypes["FrequencyScore"] > 0].copy()
pheno_filtered["Source"]     = "orphanet"
print(f"  Done: {len(pheno_filtered):,} rows | {pheno_filtered['OrphaCode'].nunique():,} rare diseases")
print(f"  Unique HPO symptoms: {pheno_filtered['HPO_Term'].nunique():,}")

# ─────────────────────────────────────────────────────────────
# 5. HPOA PHENOTYPES → LONG FORMAT (if available)
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Processing HPOA phenotypes...")

hpoa_long = None
if HAS_HPOA:
    print(f"  Columns: {hpoa.columns.tolist()}")

    # Detect column names flexibly
    col_map = {c.lower(): c for c in hpoa.columns}

    disease_col = next((col_map[k] for k in ["disease_name","diseasename","name"] if k in col_map), None)
    hpo_id_col  = next((col_map[k] for k in ["hpo_id","hpoid","hp_id"] if k in col_map), None)
    hpo_nm_col  = next((col_map[k] for k in ["hpo_name","hponame","hp_name","hpo_term"] if k in col_map), None)
    freq_col    = next((col_map[k] for k in ["frequency","freq"] if k in col_map), None)
    db_col      = next((col_map[k] for k in ["database_id","db_id","disease_id"] if k in col_map), None)

    if disease_col and (hpo_id_col or hpo_nm_col):
        hpoa_long = pd.DataFrame()
        hpoa_long["DiseaseName"]    = hpoa[disease_col].astype(str).str.strip()
        hpoa_long["HPO_ID"]         = hpoa[hpo_id_col].astype(str) if hpo_id_col else ""
        hpoa_long["HPO_Term"]       = hpoa[hpo_nm_col].astype(str).str.strip() if hpo_nm_col else hpoa_long["HPO_ID"]
        hpoa_long["FrequencyScore"] = 0.5   # default — HPOA doesn't always have frequency
        hpoa_long["Source"]         = "hpoa"
        hpoa_long["OrphaCode"]      = None

        # Try to extract ORPHA code from database_id (e.g. "ORPHA:58")
        if db_col:
            orpha_mask = hpoa[db_col].astype(str).str.startswith("ORPHA:")
            hpoa_long.loc[orpha_mask, "OrphaCode"] = (
                hpoa.loc[orpha_mask, db_col].astype(str)
                .str.replace("ORPHA:","").str.strip()
            )

        # Map frequency text to score if column exists
        if freq_col:
            hpoa_freq_map = {
                "HP:0040280": 1.0,  # Obligate
                "HP:0040281": 0.9,  # Very frequent
                "HP:0040282": 0.6,  # Frequent
                "HP:0040283": 0.3,  # Occasional
                "HP:0040284": 0.1,  # Very rare
                "HP:0040285": 0.0,  # Excluded
            }
            hpoa_long["FrequencyScore"] = hpoa[freq_col].map(hpoa_freq_map).fillna(0.5)
            hpoa_long = hpoa_long[hpoa_long["FrequencyScore"] > 0]

        hpoa_long = hpoa_long[["OrphaCode","DiseaseName","HPO_ID","HPO_Term","FrequencyScore","Source"]]
        print(f"  Done: {len(hpoa_long):,} rows | {hpoa_long['DiseaseName'].nunique():,} diseases")
    else:
        print(f"  ⚠️ Could not map columns — skipping HPOA")
        hpoa_long = None
else:
    print("  Skipped (file not found)")

# ─────────────────────────────────────────────────────────────
# 6. BUILD UNIFIED DISEASE METADATA TABLE
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Building disease metadata...")

# Merge complete + info to get all cross-reference codes
info_cols = ["OrphaCode","MONDO","UMLS","MeSH","MedDRA","GARD"]
info_clean = info[[c for c in info_cols if c in info.columns]].drop_duplicates("OrphaCode")

meta = complete[[
    "OrphaCode","DiseaseName","DisorderType","DisorderGroup","ICD-10","ICD-11","OMIM"
]].copy()

# Add extra codes from info file
meta = meta.merge(info_clean, on="OrphaCode", how="left")

# Add onset and inheritance
nat_clean = nat_history[["OrphaCode","AgeOfOnset","TypeOfInheritance"]].drop_duplicates("OrphaCode")
meta      = meta.merge(nat_clean, on="OrphaCode", how="left")

# Add prevalence
prev_clean = (prevalence[prevalence["PrevalenceClass"].notna()]
              .sort_values("PrevalenceClass")
              .drop_duplicates("OrphaCode")[["OrphaCode","PrevalenceClass"]])
meta       = meta.merge(prev_clean, on="OrphaCode", how="left")

# Add gene info
genes_agg = (genes.groupby("OrphaCode").agg(
    GeneSymbols=("GeneSymbol", lambda x: "|".join(x.dropna().unique())),
    GeneCount=("GeneSymbol","nunique")).reset_index())
meta      = meta.merge(genes_agg, on="OrphaCode", how="left")

# Add descriptions and precautions
meta = meta.merge(kaggle_desc.rename(columns={"Disease":"DiseaseName"}), on="DiseaseName", how="left")
meta = meta.merge(kaggle_precaution.rename(columns={"Disease":"DiseaseName"}), on="DiseaseName", how="left")

# Fill missing
meta["AgeOfOnset"]        = meta["AgeOfOnset"].fillna("Unknown")
meta["TypeOfInheritance"] = meta["TypeOfInheritance"].fillna("Unknown")
meta["PrevalenceClass"]   = meta["PrevalenceClass"].fillna("Unknown")
meta["GeneSymbols"]       = meta["GeneSymbols"].fillna("Unknown")
meta["GeneCount"]         = meta["GeneCount"].fillna(0).astype(int)
meta["DiseaseType"]       = "rare"

# Add Kaggle-only common diseases
all_kaggle   = set(kaggle_long["DiseaseName"].unique()) | set(profile_long["DiseaseName"].unique())
orpha_names  = set(meta["DiseaseName"].dropna().unique())
common_only  = all_kaggle - orpha_names
if common_only:
    common_meta = pd.DataFrame({
        "DiseaseName": list(common_only), "DiseaseType": "common",
        "AgeOfOnset": "Unknown","TypeOfInheritance": "Unknown",
        "PrevalenceClass": "Unknown","GeneCount": 0,"GeneSymbols": "Unknown"
    })
    meta = pd.concat([meta, common_meta], ignore_index=True)

print(f"  Metadata columns: {meta.columns.tolist()}")
print(f"  Total: {len(meta):,} diseases ({(meta['DiseaseType']=='rare').sum():,} rare | {(meta['DiseaseType']=='common').sum()} common)")

# ─────────────────────────────────────────────────────────────
# 7. COMBINE ALL SOURCES → LONG FORMAT
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Combining all symptom sources...")

sources = [
    pheno_filtered[["OrphaCode","DiseaseName","HPO_ID","HPO_Term","FrequencyScore","Source"]],
    kaggle_long[["OrphaCode","DiseaseName","HPO_Term","FrequencyScore","Source"]],
    profile_long[["OrphaCode","DiseaseName","HPO_Term","FrequencyScore","Source"]],
]
if hpoa_long is not None:
    sources.append(hpoa_long[["OrphaCode","DiseaseName","HPO_Term","FrequencyScore","Source"]])
    print(f"  Including HPOA data ✅")

combined = pd.concat(sources, ignore_index=True)
combined.to_csv(out_path + "dataset_long.csv", index=False)
print(f"  dataset_long.csv: {len(combined):,} rows | {combined['DiseaseName'].nunique():,} diseases")

# ─────────────────────────────────────────────────────────────
# 8. BUILD WIDE FORMAT ML MATRIX
# ─────────────────────────────────────────────────────────────
print("\n[8/8] Building ML training matrix...")

pivot_df = combined.copy()
pivot_df["HPO_Term_clean"] = (
    pivot_df["HPO_Term"].str.strip().str.lower()
    .str.replace(" ","_",regex=False)
    .str.replace("/","_",regex=False)
    .str.replace("(","",regex=False)
    .str.replace(")","",regex=False)
)

pivot = pivot_df.pivot_table(
    index="DiseaseName",
    columns="HPO_Term_clean",
    values="FrequencyScore",
    aggfunc="max"
).fillna(0)
pivot.columns = [f"symptom_{c}" for c in pivot.columns]
pivot = pivot.reset_index()

wide_df = meta[[
    "DiseaseName","OrphaCode","DiseaseType","DisorderType",
    "AgeOfOnset","TypeOfInheritance","PrevalenceClass","GeneCount"
]].merge(pivot, on="DiseaseName", how="inner")

wide_df.to_csv(out_path + "dataset_wide.csv", index=False)
meta.to_csv(out_path + "disease_metadata.csv", index=False)

symptom_count = wide_df.shape[1] - 8
print(f"  dataset_wide.csv: {wide_df.shape[0]:,} diseases × {wide_df.shape[1]:,} columns")
print(f"  Symptom feature columns: {symptom_count:,}")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE COMPLETE — SUMMARY")
print("="*60)
print(f"  Total diseases in ML matrix  : {wide_df.shape[0]:,}")
print(f"    Rare diseases  (Orphanet)  : {(wide_df['DiseaseType']=='rare').sum():,}")
print(f"    Common diseases (Kaggle)   : {(wide_df['DiseaseType']=='common').sum():,}")
print(f"  Total symptom features       : {symptom_count:,}")
print(f"  Avg symptoms per disease     : {(wide_df.iloc[:,8:]>0).sum(axis=1).mean():.1f}")
print(f"  Diseases with gene data      : {(wide_df['GeneCount']>0).sum():,}")
print(f"  Diseases with prevalence     : {(wide_df['PrevalenceClass']!='Unknown').sum():,}")
print(f"  HPOA data included           : {'Yes ✅' if HAS_HPOA else 'No — add hpoa_phenotypes.csv to data/raw/ to include'}")
print(f"\nSaved to data/processed/:")
print(f"  dataset_wide.csv       ← USE THIS FOR MODEL TRAINING")
print(f"  dataset_long.csv       ← full symptom-disease rows")
print(f"  disease_metadata.csv   ← all disease info + codes + genes")
print(f"\nNext step: python3 src/train_model.py")
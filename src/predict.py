import pandas as pd
import numpy as np
import joblib
import os
import gc

print("Loading RARE-AI model...")

model_path = "models/"

# Load one at a time and free memory between loads
le           = joblib.load(model_path + "label_encoder.pkl")
symptom_cols = joblib.load(model_path + "symptom_columns.pkl")
model        = joblib.load(model_path + "rare_ai_model.pkl")
gc.collect()

metadata = pd.read_csv("data/processed/disease_metadata.csv", low_memory=False)
print(f"Model ready — {len(le.classes_):,} diseases, {len(symptom_cols):,} symptoms\n")

def normalize(s):
    return s.strip().lower().replace(" ","_").replace("/","_").replace("(","").replace(")","")

def predict_diseases(symptoms, top_n=10):
    feature_vector = np.zeros(len(symptom_cols), dtype=np.float32)
    matched, unmatched = [], []

    for symptom in symptoms:
        norm = normalize(symptom)
        col  = f"symptom_{norm}"
        if col in symptom_cols:
            feature_vector[symptom_cols.index(col)] = 1.0
            matched.append(symptom)
        else:
            found = False
            for col_name in symptom_cols:
                if norm in col_name:
                    feature_vector[symptom_cols.index(col_name)] = 0.8
                    matched.append(f"{symptom}*")
                    found = True
                    break
            if not found:
                unmatched.append(symptom)

    print(f"  Matched   : {matched}")
    if unmatched:
        print(f"  Not found : {unmatched} (try rephrasing)")

    if not matched:
        print("  No symptoms matched.")
        return pd.DataFrame()

    proba           = model.predict_proba(feature_vector.reshape(1,-1))[0]
    trained_classes = model.classes_
    top_idx         = np.argsort(proba)[::-1][:top_n]
    top_names       = le.inverse_transform(trained_classes[top_idx])
    top_confs       = (proba[top_idx] * 100).round(3)

    results = pd.DataFrame({"Rank": range(1, top_n+1),
                             "DiseaseName": top_names,
                             "Confidence":  top_confs})

    meta_cols = [c for c in ["DiseaseName","OrphaCode","DiseaseType","AgeOfOnset",
                              "TypeOfInheritance","PrevalenceClass","GeneSymbols",
                              "Description"] if c in metadata.columns]
    results = results.merge(
        metadata[meta_cols].drop_duplicates("DiseaseName"),
        on="DiseaseName", how="left"
    )
    return results


def print_predictions(results, symptoms):
    print("\n" + "="*65)
    print("RARE-AI — PREDICTION RESULTS")
    print("="*65)
    print(f"Symptoms: {', '.join(symptoms)}\n")

    for _, row in results.iterrows():
        tag   = "🔴 RARE" if row.get("DiseaseType") == "rare" else "🟡 COMMON"
        orpha = f" [OrphaCode: {int(row['OrphaCode'])}]" if pd.notna(row.get("OrphaCode")) else ""
        print(f"#{int(row['Rank'])}  {row['DiseaseName']}")
        print(f"    Confidence    : {row['Confidence']}%  {tag}{orpha}")
        if pd.notna(row.get("AgeOfOnset"))        and row["AgeOfOnset"] != "Unknown":
            print(f"    Age of Onset  : {row['AgeOfOnset']}")
        if pd.notna(row.get("TypeOfInheritance")) and row["TypeOfInheritance"] != "Unknown":
            print(f"    Inheritance   : {row['TypeOfInheritance']}")
        if pd.notna(row.get("PrevalenceClass"))   and row["PrevalenceClass"] != "Unknown":
            print(f"    Prevalence    : {row['PrevalenceClass']}")
        genes = str(row.get("GeneSymbols",""))
        if genes not in ("Unknown","nan",""):
            print(f"    Genes         : {', '.join(genes.split('|')[:3])}")
        print()

    print("⚠️  Second-opinion tool only — confirm with a specialist.")
    print("="*65)


if __name__ == "__main__":
    # Test 1 — rare neurological
    print("📋 Test 1: Rare neurological")
    r1 = predict_diseases(["macrocephaly","seizures","intellectual disability"], top_n=5)
    print_predictions(r1, ["macrocephaly","seizures","intellectual disability"])

    # Test 2 — common
    print("📋 Test 2: Common symptoms")
    r2 = predict_diseases(["fever","cough","fatigue"], top_n=5)
    print_predictions(r2, ["fever","cough","fatigue"])

    # Interactive
    user = input("\nEnter your own symptoms (comma separated): ")
    syms = [s.strip() for s in user.split(",") if s.strip()]
    r3   = predict_diseases(syms, top_n=10)
    print_predictions(r3, syms)
    if not r3.empty:
        r3.to_csv("data/processed/last_prediction.csv", index=False)
        print("Saved → data/processed/last_prediction.csv")
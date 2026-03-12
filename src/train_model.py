import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

print("=" * 60)
print("RARE-AI: Model Training Pipeline")
print("=" * 60)

data_path  = "data/processed/"
model_path = "models/"
os.makedirs(model_path, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading processed dataset...")

df = pd.read_csv(data_path + "dataset_wide.csv", low_memory=False)
print(f"  Loaded: {df.shape[0]:,} diseases × {df.shape[1]:,} columns")
print(f"  Rare: {(df['DiseaseType']=='rare').sum():,} | Common: {(df['DiseaseType']=='common').sum():,}")

# ─────────────────────────────────────────────────────────────
# 2. PREPARE FEATURES AND LABELS
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Preparing features and labels...")

symptom_cols = [c for c in df.columns if c.startswith("symptom_")]
X_full = df[symptom_cols].values.astype(np.float32)

# Feature reduction — keep only symptoms that appear in 3+ diseases
# This cuts ~50% of features while keeping all meaningful ones
print(f"  Raw symptom features: {len(symptom_cols):,}")
symptom_presence = (X_full > 0).sum(axis=0)
keep_mask = symptom_presence >= 3
X = X_full[:, keep_mask]
kept_symptom_cols = [symptom_cols[i] for i in range(len(symptom_cols)) if keep_mask[i]]
print(f"  After filtering (appear in 3+ diseases): {X.shape[1]:,} features")
print(f"  Memory: {X.nbytes / 1024 / 1024:.1f} MB")

# Encode disease labels
le = LabelEncoder()
y = le.fit_transform(df["DiseaseName"].values)
print(f"  Number of disease classes: {len(le.classes_):,}")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Splitting into train/test sets (80/20)...")

disease_type = df["DiseaseType"].values
X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(
    X, y, disease_type,
    test_size=0.2,
    random_state=42,
    stratify=disease_type
)

print(f"  Train: {X_train.shape[0]:,} diseases")
print(f"  Test:  {X_test.shape[0]:,} diseases")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Training Random Forest classifier...")
print("  (This takes 3-8 minutes on first run — please wait...)")

model = RandomForestClassifier(
    n_estimators=200,         # 200 decision trees
    max_depth=None,           # full depth — important for rare diseases
    min_samples_leaf=1,       # allows learning from single examples
    max_features="sqrt",      # standard best practice
    class_weight="balanced",  # corrects for rare disease imbalance
    random_state=42,
    n_jobs=-1,                # uses all your CPU cores
    verbose=1
)

model.fit(X_train, y_train)
print("\n  Training complete.")

# ─────────────────────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Evaluating model...")

y_pred   = model.predict(X_test)
y_proba  = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)

# Manual top-k: works correctly when not all classes appear in test set
trained_classes = model.classes_

def top_k_acc(y_true, y_proba, k):
    correct = 0
    for true_label, probs in zip(y_true, y_proba):
        top_k_labels = trained_classes[np.argsort(probs)[::-1][:k]]
        if true_label in top_k_labels:
            correct += 1
    return correct / len(y_true)

top5  = top_k_acc(y_test, y_proba, k=5)
top10 = top_k_acc(y_test, y_proba, k=10)

print(f"\n  ┌─────────────────────────────────┐")
print(f"  │  Exact match accuracy:  {acc*100:6.2f}%  │")
print(f"  │  Top-5  accuracy:       {top5*100:6.2f}%  │")
print(f"  │  Top-10 accuracy:       {top10*100:6.2f}%  │")
print(f"  └─────────────────────────────────┘")

# Accuracy split by disease type
rare_mask   = dt_test == "rare"
common_mask = dt_test == "common"

if rare_mask.sum() > 0:
    rare_acc = accuracy_score(y_test[rare_mask], y_pred[rare_mask])
    print(f"\n  Rare disease accuracy:    {rare_acc*100:.2f}%  ({rare_mask.sum()} test diseases)")

if common_mask.sum() > 0:
    common_acc = accuracy_score(y_test[common_mask], y_pred[common_mask])
    print(f"  Common disease accuracy:  {common_acc*100:.2f}%  ({common_mask.sum()} test diseases)")

# Top 15 most important symptoms
print(f"\n  Top 15 most predictive symptoms:")
fi = pd.Series(model.feature_importances_, index=kept_symptom_cols).sort_values(ascending=False)
for sym, score in fi.head(15).items():
    name = sym.replace("symptom_", "").replace("_", " ")
    print(f"    {name:<45} {score:.4f}")

# ─────────────────────────────────────────────────────────────
# 6. SAVE MODEL AND ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Saving model and artifacts...")

joblib.dump(model,             model_path + "rare_ai_model.pkl")
joblib.dump(le,                model_path + "label_encoder.pkl")
joblib.dump(kept_symptom_cols, model_path + "symptom_columns.pkl")

fi.reset_index().rename(columns={"index": "symptom", 0: "importance"}).to_csv(
    model_path + "feature_importances.csv", index=False
)

pd.DataFrame([{
    "exact_accuracy":   round(acc * 100, 2),
    "top5_accuracy":    round(top5 * 100, 2),
    "top10_accuracy":   round(top10 * 100, 2),
    "total_diseases":   int(len(le.classes_)),
    "symptom_features": int(len(kept_symptom_cols)),
    "train_size":       int(X_train.shape[0]),
    "test_size":        int(X_test.shape[0]),
}]).to_csv(model_path + "eval_summary.csv", index=False)

print(f"  Saved → models/rare_ai_model.pkl")
print(f"  Saved → models/label_encoder.pkl")
print(f"  Saved → models/symptom_columns.pkl")
print(f"  Saved → models/feature_importances.csv")
print(f"  Saved → models/eval_summary.csv")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Diseases in model:     {len(le.classes_):,}")
print(f"  Symptom features:      {len(kept_symptom_cols):,}")
print(f"  Exact accuracy:        {acc*100:.2f}%")
print(f"  Top-5 accuracy:        {top5*100:.2f}%")
print(f"  Top-10 accuracy:       {top10*100:.2f}%")
print(f"\nNext step: python3 src/predict.py")
"""
Run this once to rebuild a lighter model that loads faster.
Replace src/reduce_model.py and run: python3 src/reduce_model.py
"""
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("RARE-AI: Building lightweight model...")

data_path  = "data/processed/"
model_path = "models/"
os.makedirs(model_path, exist_ok=True)

df = pd.read_csv(data_path + "dataset_wide.csv", low_memory=False)
symptom_cols = [c for c in df.columns if c.startswith("symptom_")]

X_full = df[symptom_cols].values.astype(np.float32)

# Keep only symptoms that appear in 5+ diseases (more aggressive reduction)
presence   = (X_full > 0).sum(axis=0)
keep_mask  = presence >= 5
X          = X_full[:, keep_mask]
kept_cols  = [symptom_cols[i] for i in range(len(symptom_cols)) if keep_mask[i]]

print(f"  Features reduced: {len(symptom_cols):,} → {X.shape[1]:,}")
print(f"  Memory: {X.nbytes/1024/1024:.1f} MB")

le = LabelEncoder()
y  = le.fit_transform(df["DiseaseName"].values)
disease_type = df["DiseaseType"].values

X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(
    X, y, disease_type, test_size=0.2, random_state=42, stratify=disease_type
)

print(f"  Training {len(le.classes_):,} diseases with {X_train.shape[1]:,} features...")
print("  (Using 100 trees for lighter model — takes ~2 min)")

model = RandomForestClassifier(
    n_estimators=100,         # reduced from 200 — still accurate, half the size
    max_depth=30,             # cap depth — saves memory significantly
    min_samples_leaf=2,       # slight smoothing
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

# Manual top-5
trained_classes = model.classes_
y_proba = model.predict_proba(X_test)

def top_k_acc(y_true, y_proba, k):
    correct = 0
    for true_label, probs in zip(y_true, y_proba):
        top_k_labels = trained_classes[np.argsort(probs)[::-1][:k]]
        if true_label in top_k_labels:
            correct += 1
    return correct / len(y_true)

top5  = top_k_acc(y_test, y_proba, k=5)
top10 = top_k_acc(y_test, y_proba, k=10)

print(f"\n  Exact accuracy: {acc*100:.2f}%")
print(f"  Top-5 accuracy: {top5*100:.2f}%")
print(f"  Top-10 accuracy: {top10*100:.2f}%")

# Save with compression to reduce file size
print("\n  Saving compressed model...")
joblib.dump(model,      model_path + "rare_ai_model.pkl",      compress=3)
joblib.dump(le,         model_path + "label_encoder.pkl",      compress=3)
joblib.dump(kept_cols,  model_path + "symptom_columns.pkl",    compress=3)

# Save feature importances
fi = pd.Series(model.feature_importances_, index=kept_cols).sort_values(ascending=False)
fi.reset_index().to_csv(model_path + "feature_importances.csv", index=False)

pd.DataFrame([{
    "exact_accuracy":   round(acc*100,2),
    "top5_accuracy":    round(top5*100,2),
    "top10_accuracy":   round(top10*100,2),
    "total_diseases":   int(len(le.classes_)),
    "symptom_features": int(len(kept_cols)),
}]).to_csv(model_path + "eval_summary.csv", index=False)

import os
size_mb = os.path.getsize(model_path + "rare_ai_model.pkl") / 1024 / 1024
print(f"\n  Model size: {size_mb:.1f} MB")
print("  Done — run predict.py now")
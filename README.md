# 🧬 RARE-AI

## Machine Learning System for Early Detection of Rare Diseases

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)

RARE-AI is a **machine learning–powered healthcare decision support system** designed to assist in the **early detection of rare and hard-to-diagnose diseases** using patient symptoms and clinical datasets.

The system combines **rare disease knowledge bases, symptom datasets, and machine learning models** to predict possible rare diseases and assist clinicians in identifying high-risk cases earlier.

---

# 📌 Project Motivation

Rare diseases affect millions of people worldwide but are often **misdiagnosed due to complex and overlapping symptoms**.

Patients frequently experience a **diagnostic odyssey**, where it takes years to identify the correct condition.

RARE-AI aims to address this problem by using **machine learning to detect hidden symptom patterns** that may indicate rare diseases.

---

# ⚙️ Key Features

🧠 **Machine Learning Prediction Engine**
Predicts rare diseases based on symptom patterns.

📊 **Multi-Dataset Integration**
Combines datasets from Orphanet, HPO, and Kaggle.

📉 **Feature Importance Analysis**
Identifies the most influential symptoms.

🌐 **Web Interface for Predictions**
Built using Flask with custom HTML and CSS.

📁 **Modular Machine Learning Pipeline**
Includes preprocessing, model training, prediction, and evaluation.

📦 **Model Persistence**
Trained models saved using Joblib.

---

# 🏗️ System Architecture

```id="qv6h4t"
Patient Symptoms
       │
       ▼
Web Interface (Flask)
       │
       ▼
Feature Encoding
       │
       ▼
Machine Learning Model (XGBoost)
       │
       ▼
Rare Disease Prediction
       │
       ▼
Confidence Score + Explanation
```

---

# 📂 Project Structure

```id="1g1sp5"
RARE-AI/
│
├── data/
│   ├── raw/                     # Original datasets
│   │   ├── dataset.csv
│   │   ├── Disease_symptom_and_patient_data.csv
│   │   ├── hpoa_phenotypes.csv
│   │   ├── rare_diseases_complete.csv
│   │   ├── rare_diseases_genes.csv
│   │   ├── rare_diseases_info.csv
│   │   ├── rare_diseases_natural_history.csv
│   │   ├── rare_diseases_prevalence.csv
│   │   ├── symptom_Description.csv
│   │   ├── symptom_precaution.csv
│   │   └── Symptom-severity.csv
│   │
│   └── processed/               # Cleaned datasets
│       ├── dataset_long.csv
│       ├── dataset_wide.csv
│       ├── disease_metadata.csv
│       └── last_prediction.csv
│
├── models/
│   ├── rare_ai_model.pkl        # Trained ML model
│   ├── label_encoder.pkl
│   ├── symptom_columns.pkl
│   ├── feature_importances.csv
│   └── eval_summary.csv
│
├── src/
│   ├── app.py                   # Flask web application
│   ├── predict.py               # Prediction logic
│   ├── prepare_data.py          # Dataset preprocessing
│   ├── train_model.py           # Model training pipeline
│   └── reduce_model.py          # Model optimization
│
├── templates/
│   └── index.html               # Web interface
│
├── static/                      # CSS & frontend assets
│
├── File Converter/
│   ├── hpoa_to_csv.py
│   └── xml_to_csv.py
│
├── Report/
│   ├── Individual_Project.pdf
│   ├── Project_Report.pdf
│   └── Supporting reports
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📊 Datasets Used

The project integrates multiple datasets to build a comprehensive rare disease prediction system.

### 📚 Data Sources

| Dataset                    | Description                               |
| -------------------------- | ----------------------------------------- |
| Rare Disease Dataset       | Core dataset of rare disease information  |
| HPO Phenotypes             | Human Phenotype Ontology symptom mapping  |
| Disease-Symptom Dataset    | Mapping of diseases to symptoms           |
| Symptom Severity Dataset   | Severity ranking of symptoms              |
| Symptom Precaution Dataset | Recommended medical precautions           |
| Rare Disease Metadata      | Prevalence, genetics, and natural history |

These datasets are **merged and processed** to create a structured dataset for training the machine learning model.

---

# 🤖 Machine Learning Model

The project uses **XGBoost**, a gradient boosting algorithm widely used for structured medical data.

### Model Pipeline

1️⃣ Data preprocessing
2️⃣ Symptom encoding
3️⃣ Feature extraction
4️⃣ Train-test split
5️⃣ Model training
6️⃣ Model evaluation
7️⃣ Model saving

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

Model evaluation results are stored in:

```id="4z7t9u"
models/eval_summary.csv
```

---

# 🧪 Installation

### 1️⃣ Clone the Repository

```id="h0jeh2"
git clone https://github.com/paryej07/RARE-AI.git
cd RARE-AI
```

---

### 2️⃣ Install Dependencies

```id="c28k7s"
pip install -r requirements.txt
```

---

# 📊 Data Preparation

Run the preprocessing pipeline:

```id="d9eyfo"
python src/prepare_data.py
```

This generates processed datasets inside:

```id="9ok7nv"
data/processed/
```

---

# 🧠 Train the Model

```id="ru53k1"
python src/train_model.py
```

The trained model will be saved inside:

```id="08d6z8"
models/
```

---

# 🌐 Run the Web Application

```id="0hsb5y"
python src/app.py
```

Open in browser:

```id="1l7x5y"
http://127.0.0.1:5000
```

---

# 🧪 Example Prediction

### Input

```id="aqe8hp"
fatigue, muscle weakness, joint pain
```

### Output

```id="kpskhb"
Predicted Disease: Myasthenia Gravis
Confidence Score: 86%
```

---

# 🚀 Future Improvements

🔬 Deep learning models for rare disease prediction
📈 Explainable AI (SHAP / LIME)
☁️ Cloud deployment
🏥 Integration with Electronic Health Records (EHR)
📊 Clinical decision support dashboards

---

# ⚠️ Disclaimer

This project is intended **for educational and research purposes only**.
It should **not be used as a substitute for professional medical diagnosis**.

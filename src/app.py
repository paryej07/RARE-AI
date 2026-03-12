from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import sys, os, re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

from predict import symptom_cols, model, le, metadata, normalize

# ── Symptom phrase lookup (longest first for greedy matching) ──
symptom_phrases = sorted(
    [col.replace("symptom_", "").replace("_", " ") for col in symptom_cols],
    key=lambda x: -len(x)
)

STOPWORDS = {
    "i","have","had","has","am","is","are","was","were","be","been","a","an",
    "the","and","or","but","with","for","from","since","past","last","days",
    "weeks","months","years","day","week","month","my","me","its","it","very",
    "quite","bit","little","some","also","too","feel","feeling","experiencing",
    "suffering","noticed","getting","got","give","two","three","four","five",
    "of","in","at","on","when","that","this","these","those","about","like",
    "just","not","no","so","do","did","can","could","would","there","him",
    "her","they","we","you","your","doctor","hospital","please","help",
    "severe","mild","moderate","chronic","acute","sudden","recent","constant",
    "intermittent","occasional","always","often","sometimes","also","already"
}

def extract_symptoms_from_text(text: str) -> list:
    text_clean = re.sub(r"[^\w\s\-]", " ", text.lower())
    found, found_positions = [], set()

    for phrase in symptom_phrases:
        for match in re.finditer(r'\b' + re.escape(phrase) + r'\b', text_clean):
            positions = set(range(match.start(), match.end()))
            if not positions & found_positions:
                found.append(phrase)
                found_positions |= positions

    # Fallback: unmatched words
    remaining = text_clean
    for pos in sorted(found_positions, reverse=True):
        remaining = remaining[:pos] + " " + remaining[pos+1:]
    for word in remaining.split():
        if word not in STOPWORDS and len(word) > 3:
            for phrase in symptom_phrases:
                if word in phrase.split() and phrase not in found:
                    found.append(phrase)
                    break

    return list(dict.fromkeys(found))


def build_prediction(symptoms_input, top_n=10):
    matched, unmatched = [], []
    fv = np.zeros(len(symptom_cols), dtype=np.float32)

    for s in symptoms_input:
        norm = normalize(s)
        col  = f"symptom_{norm}"
        if col in symptom_cols:
            fv[symptom_cols.index(col)] = 1.0
            matched.append(s)
        else:
            found = False
            for c in symptom_cols:
                if norm in c:
                    fv[symptom_cols.index(c)] = 0.8
                    matched.append(s)
                    found = True
                    break
            if not found:
                unmatched.append(s)

    if not matched:
        return None, matched, unmatched

    proba  = model.predict_proba(fv.reshape(1, -1))[0]
    tc     = model.classes_
    top_i  = np.argsort(proba)[::-1][:top_n]
    names  = le.inverse_transform(tc[top_i])
    confs  = (proba[top_i] * 100).round(2)

    results = pd.DataFrame({"DiseaseName": names, "Confidence": confs})
    mc = ["DiseaseName","OrphaCode","DiseaseType","AgeOfOnset",
          "TypeOfInheritance","PrevalenceClass","GeneSymbols","Description"]
    avail = [c for c in mc if c in metadata.columns]
    results = results.merge(
        metadata[avail].drop_duplicates("DiseaseName"),
        on="DiseaseName", how="left"
    )
    return results, matched, unmatched


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json()
    symptoms = data.get("symptoms", [])
    top_n    = data.get("top_n", 10)
    patient  = data.get("patient", {})

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    results, matched, unmatched = build_prediction(symptoms, top_n)
    if results is None:
        return jsonify({"predictions":[], "matched":[], "unmatched": unmatched, "patient": patient})

    return jsonify({
        "predictions": results.fillna("").to_dict(orient="records"),
        "matched":     matched,
        "unmatched":   unmatched,
        "patient":     patient
    })


@app.route("/parse", methods=["POST"])
def parse():
    data    = request.get_json()
    text    = data.get("text", "").strip()
    top_n   = data.get("top_n", 10)
    patient = data.get("patient", {})

    if not text:
        return jsonify({"error": "No text provided"}), 400

    extracted = extract_symptoms_from_text(text)
    if not extracted:
        return jsonify({
            "extracted":[], "predictions":[], "matched":[], "unmatched":[],
            "patient": patient,
            "message": "No recognisable symptoms found. Try using medical terms like 'fever', 'joint pain', 'fatigue'."
        })

    results, matched, unmatched = build_prediction(extracted, top_n)
    if results is None:
        return jsonify({"extracted": extracted, "predictions":[], "matched":[], "unmatched": unmatched, "patient": patient})

    return jsonify({
        "extracted":   extracted,
        "predictions": results.fillna("").to_dict(orient="records"),
        "matched":     matched,
        "unmatched":   unmatched,
        "patient":     patient
    })


@app.route("/symptoms", methods=["GET"])
def list_symptoms():
    clean = [c.replace("symptom_","").replace("_"," ") for c in symptom_cols]
    return jsonify({"symptoms": clean, "count": len(clean)})


if __name__ == "__main__":
    print("=" * 55)
    print("RARE-AI Web App")
    print("Server: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("=" * 55)
    app.run(debug=False, port=5000)
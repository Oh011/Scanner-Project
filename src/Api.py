"""
api.py  —  Java Vulnerability Scanner
FastAPI endpoint with a clean, explicit response shape.

Response shape (POST /predict):
{
  "file": "BenchmarkTest00001.java",
  "verdict": {
    "is_vulnerable":          true,
    "vulnerable_probability": 0.87,
    "safe_probability":       0.13
  },
  "category": {                        // null if not vulnerable
    "predicted":    "pathtraver",
    "confidence":   0.92,
    "probabilities": { "pathtraver": 0.92, "sqli": 0.04, ... }
  },
  "features": {
    "values": { "taint_source_count": 3, ... },
    "importances": [
      { "feature": "total_chars", "importance": 0.29 },
      ...
    ]
  }
}
"""

import os, sys, numpy as np, pandas as pd, joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.scanners.java_features import extract_features



from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BINARY_MODEL = BASE_DIR / "src" / "models" / "sklearn_gb_model_2.pkl"
CATEGORY_MODEL = BASE_DIR / "src" / "models" / "gb_category_classification.pkl"

app = FastAPI(title="Java Vulnerability Scanner", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

binary_model = binary_feature_columns = None
category_model = category_feature_columns = category_classes = None


@app.on_event("startup")
def load_models():
    global binary_model, binary_feature_columns
    global category_model, category_feature_columns, category_classes
    bp = joblib.load(BINARY_MODEL)
    binary_model, binary_feature_columns = bp["model"], bp["feature_columns"]
    cp = joblib.load(CATEGORY_MODEL)
    category_model = cp["model"]
    category_feature_columns = cp["feature_columns"]
    category_classes = cp["classes"]
    print("Models loaded.")


def _row(features, columns):
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return pd.DataFrame([{c: features[c] for c in columns}])


def _importances(model, columns):
    pairs = sorted(zip(columns, model.feature_importances_.tolist()),
                   key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(v, 6)} for f, v in pairs]


@app.get("/health")
def health():
    return {"status": "ok",
            "binary_loaded": binary_model is not None,
            "category_loaded": category_model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if binary_model is None:
        raise HTTPException(503, "Models not loaded.")
    if not file.filename.endswith(".java"):
        raise HTTPException(400, "Only .java files are accepted.")

    source_code = (await file.read()).decode("utf-8", errors="ignore")
    if not source_code.strip():
        raise HTTPException(400, "File is empty.")

    try:
        features = extract_features(source_code, file_name=file.filename)
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    # Binary
    try:
        X_bin = _row(features, binary_feature_columns)
        bin_pred = int(binary_model.predict(X_bin)[0])
        bin_proba = binary_model.predict_proba(X_bin)[0]
        safe_prob = round(float(bin_proba[0]), 4)
        vuln_prob = round(float(bin_proba[1]), 4)
    except Exception as e:
        raise HTTPException(500, f"Binary prediction failed: {e}")

    # Category (cascade)
    category_payload = None
    if bin_pred == 1:
        try:
            X_cat = _row(features, category_feature_columns)
            cat_proba = category_model.predict_proba(X_cat)[0]
            best_idx = int(np.argmax(cat_proba))
            class_probs = {
                cls: round(float(p), 4)
                for cls, p in sorted(zip(category_classes, cat_proba.tolist()),
                                     key=lambda x: x[1], reverse=True)
            }
            category_payload = {
                "predicted":     category_classes[best_idx],
                "confidence":    round(float(cat_proba[best_idx]), 4),
                "probabilities": class_probs,
            }
        except Exception as e:
            raise HTTPException(500, f"Category prediction failed: {e}")

    feature_values = {k: v for k, v in features.items()
                      if k != "file_name" and not isinstance(v, str)}

    return JSONResponse({
        "file": file.filename,
        "verdict": {
            "is_vulnerable":          bool(bin_pred),
            "vulnerable_probability": vuln_prob,
            "safe_probability":       safe_prob,
        },
        "category": category_payload,
        "features": {
            "values":      feature_values,
            "importances": _importances(binary_model, binary_feature_columns),
        },
    })


@app.post("/features")
async def features_only(file: UploadFile = File(...)):
    if not file.filename.endswith(".java"):
        raise HTTPException(400, "Only .java files are accepted.")
    src = (await file.read()).decode("utf-8", errors="ignore")
    return JSONResponse({"file": file.filename,
                         "features": extract_features(src, file.filename)})
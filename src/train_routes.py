"""
train_routes.py
FastAPI router — mount this into your main api.py with:
    from train_routes import router as train_router
    app.include_router(train_router, prefix="/train", tags=["Training"])

Endpoints:
    POST /train/binary    — retrain the binary vulnerability model
    POST /train/category  — retrain the category classifier
    POST /train/all       — retrain both sequentially

All endpoints stream-train synchronously and return a rich metrics payload.
In production you'd run this in a background task; for a grad project
synchronous is fine since GBM on 2740 rows takes ~10 seconds.
"""

import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

router = APIRouter()

# ─── Paths ────────────────────────────────────────────────────────────────── #
BASE_DIR       = Path(__file__).resolve().parent.parent
DATASET_PATH   = BASE_DIR / "data" / "processed" / "features_2700_3.csv"
BINARY_PKL     = BASE_DIR / "src" / "models" / "sklearn_gb_model_2.pkl"
CATEGORY_PKL   = BASE_DIR / "src" / "models" / "gb_category_classification.pkl"

# ─── Column definitions (must match your training scripts) ────────────────── #
META_COLS = ["file_name", "test_name", "actual_category",
             "actual_cwe", "is_actually_vulnerable"]

LEAKAGE_COLS = [
    "sql_sink_count", "cmd_sink_count", "xss_sink_count",
    "path_sink_count", "ldap_sink_count", "xxe_sink_count",
    "weak_crypto_sink_count",
    "xss_sanitizer_count", "sql_sanitizer_count", "path_sanitizer_count",
    "generic_sanitizer_count", "total_sink_count", "total_sanitizer_count",
    "sanitizer_to_sink_ratio", "has_any_sanitizer",
    "sqli_tainted_sink_reached",  "sqli_sanitized_before_sink",
    "cmdi_tainted_sink_reached",  "cmdi_sanitized_before_sink",
    "pathtraver_tainted_sink_reached", "pathtraver_sanitized_before_sink",
    "ldapi_tainted_sink_reached", "ldapi_sanitized_before_sink",
    "xpathi_tainted_sink_reached","xpathi_sanitized_before_sink",
]


# ─── Helpers ──────────────────────────────────────────────────────────────── #

def _load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise HTTPException(404, f"Dataset not found at {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def _feature_importances(model, columns) -> list:
    pairs = sorted(zip(columns, model.feature_importances_.tolist()),
                   key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(v, 6)} for f, v in pairs]


def _per_class_report(y_true, y_pred, labels) -> dict:
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    result = {}
    for cls in labels:
        key = str(cls)
        if key in report:
            result[key] = {
                "precision": round(report[key]["precision"], 4),
                "recall":    round(report[key]["recall"],    4),
                "f1":        round(report[key]["f1-score"],  4),
                "support":   int(report[key]["support"]),
            }
    return result


# ─── Binary trainer ───────────────────────────────────────────────────────── #

def _train_binary(df: pd.DataFrame) -> dict:
    t0 = time.time()

    X = df.drop(columns=META_COLS)
    y = df["is_actually_vulnerable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save
    BINARY_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, BINARY_PKL)

    return {
        "model": "binary",
        "duration_seconds": round(time.time() - t0, 2),
        "dataset": {
            "total":      int(len(df)),
            "train":      int(len(X_train)),
            "test":       int(len(X_test)),
            "vulnerable": int((y == 1).sum()),
            "safe":       int((y == 0).sum()),
        },
        "metrics": {
            "accuracy":         round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":          round(roc_auc_score(y_test, y_proba), 4),
            "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
            "confusion_matrix": cm,   # [[TN, FP], [FN, TP]]
        },
        "feature_importances": _feature_importances(model, list(X.columns)),
        "saved_to": str(BINARY_PKL),
    }


# ─── Category trainer ─────────────────────────────────────────────────────── #

def _train_category(df: pd.DataFrame) -> dict:
    t0 = time.time()

    df_vuln = df[df["is_actually_vulnerable"] == 1].copy()

    X = df_vuln.drop(columns=LEAKAGE_COLS + META_COLS)
    y = df_vuln["actual_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    classes = list(model.classes_)

    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    # Category distribution in full vuln set
    cat_dist = df_vuln["actual_category"].value_counts().to_dict()

    # Save
    CATEGORY_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model":            model,
        "feature_columns":  list(X.columns),
        "classes":          classes,
        "version":          "2.0",
        "model_type":       "GradientBoostingClassifier",
    }, CATEGORY_PKL)

    return {
        "model": "category",
        "duration_seconds": round(time.time() - t0, 2),
        "dataset": {
            "total":              int(len(df)),
            "vulnerable":         int(len(df_vuln)),
            "train":              int(len(X_train)),
            "test":               int(len(X_test)),
            "category_distribution": {k: int(v) for k, v in cat_dist.items()},
        },
        "metrics": {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "macro_f1":  round(f1_score(y_test, y_pred, average="macro",
                                        zero_division=0), 4),
            "weighted_f1": round(f1_score(y_test, y_pred, average="weighted",
                                          zero_division=0), 4),
            "confusion_matrix": cm,
            "classes":          classes,
            "per_class":        _per_class_report(y_test, y_pred, classes),
        },
        "feature_importances": _feature_importances(model, list(X.columns)),
        "saved_to": str(CATEGORY_PKL),
    }


# ─── Routes ───────────────────────────────────────────────────────────────── #

@router.post("/binary")
def train_binary():
    """Retrain the binary vulnerability classifier. Takes ~10s."""
    df = _load_dataset()
    result = _train_binary(df)
    return JSONResponse(result)


@router.post("/category")
def train_category():
    """Retrain the vulnerability category classifier. Takes ~15s."""
    df = _load_dataset()
    result = _train_category(df)
    return JSONResponse(result)


@router.post("/all")
def train_all():
    """Retrain both models sequentially. Takes ~25s."""
    df = _load_dataset()
    binary_result   = _train_binary(df)
    category_result = _train_category(df)
    return JSONResponse({
        "binary":   binary_result,
        "category": category_result,
    })


# ─── Hyperparameter request body ─────────────────────────────────────────── #
# Add this import at the top of the file (shown here for clarity):
#   from pydantic import BaseModel
# These classes let the HTML form pass custom hyperparameters.

from pydantic import BaseModel
from typing import Optional

class HParams(BaseModel):
    n_estimators: Optional[int]  = None   # uses default if not supplied
    learning_rate: Optional[float] = None
    max_depth: Optional[int]     = None


# Override the simple routes with hyperparameter-aware versions:

@router.post("/binary/custom")
def train_binary_custom(hp: HParams):
    """Train binary model with custom hyperparameters from the UI."""
    df = _load_dataset()

    t0 = time.time()
    X = df.drop(columns=META_COLS)
    y = df["is_actually_vulnerable"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators  = hp.n_estimators  or 100,
        learning_rate = hp.learning_rate or 0.1,
        max_depth     = hp.max_depth     or 3,
        random_state  = 42,
    )
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()
    BINARY_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, BINARY_PKL)
    return JSONResponse({
        "model": "binary",
        "duration_seconds": round(time.time() - t0, 2),
        "dataset": {
            "total": int(len(df)), "train": int(len(X_train)),
            "test": int(len(X_test)),
            "vulnerable": int((y==1).sum()), "safe": int((y==0).sum()),
        },
        "metrics": {
            "accuracy":         round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":          round(roc_auc_score(y_test, y_proba), 4),
            "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
            "confusion_matrix": cm,
        },
        "feature_importances": _feature_importances(model, list(X.columns)),
        "saved_to": str(BINARY_PKL),
    })


@router.post("/category/custom")
def train_category_custom(hp: HParams):
    """Train category model with custom hyperparameters from the UI."""
    df = _load_dataset()

    t0 = time.time()
    df_vuln = df[df["is_actually_vulnerable"] == 1].copy()
    X = df_vuln.drop(columns=LEAKAGE_COLS + META_COLS)
    y = df_vuln["actual_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators  = hp.n_estimators  or 200,
        learning_rate = hp.learning_rate or 0.1,
        max_depth     = hp.max_depth     or 5,
        random_state  = 42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    classes = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    cat_dist = df_vuln["actual_category"].value_counts().to_dict()
    CATEGORY_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "feature_columns": list(X.columns),
        "classes": classes, "version": "2.0",
        "model_type": "GradientBoostingClassifier",
    }, CATEGORY_PKL)
    return JSONResponse({
        "model": "category",
        "duration_seconds": round(time.time() - t0, 2),
        "dataset": {
            "total": int(len(df)), "vulnerable": int(len(df_vuln)),
            "train": int(len(X_train)), "test": int(len(X_test)),
            "category_distribution": {k: int(v) for k, v in cat_dist.items()},
        },
        "metrics": {
            "accuracy":   round(accuracy_score(y_test, y_pred), 4),
            "macro_f1":   round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "weighted_f1":round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "confusion_matrix": cm,
            "classes": classes,
            "per_class": _per_class_report(y_test, y_pred, classes),
        },
        "feature_importances": _feature_importances(model, list(X.columns)),
        "saved_to": str(CATEGORY_PKL),
    })

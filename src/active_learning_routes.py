"""
active_learning_routes.py
FastAPI router for Active Learning + Human-in-the-Loop workflows.

Mount into api.py with:
    from active_learning_routes import router as al_router
    app.include_router(al_router, prefix="/api", tags=["Active Learning"])

Core design
-----------
Active Learning strategy: UNCERTAINTY SAMPLING
  - For each unlabelled prediction, uncertainty = 1 - max(class_proba)
  - High uncertainty ⟹ model is unsure ⟹ analyst review most valuable

Human-in-the-Loop pipeline:
  1.  POST /api/queue           — scan a .java file, compute uncertainty,
                                   push into the review queue (SQLite)
  2.  GET  /api/queue           — fetch pending reviews sorted by uncertainty
  3.  POST /api/reviews         — analyst submits a label + optional category
  4.  GET  /api/reviews         — list completed reviews with stats
  5.  POST /api/retrain-active  — retrain both models on original dataset
                                   PLUS all accepted analyst labels
  6.  GET  /api/overview        — aggregate stats (queue size, review count,
                                   model accuracy cache, category distribution)

Storage: SQLite file at  data/active_learning.db  (auto-created).
In production swap for Postgres; the SQL is ANSI-compatible.
"""

import time
import json
import sqlite3
import warnings
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

router = APIRouter()

# ─── Paths ────────────────────────────────────────────────────────────────── #
BASE_DIR       = Path(__file__).resolve().parent.parent
DATASET_PATH   = BASE_DIR / "data" / "processed" / "features_2700_3.csv"
BINARY_PKL     = BASE_DIR / "src" / "models" / "sklearn_gb_model_2.pkl"
CATEGORY_PKL   = BASE_DIR / "src" / "models" / "gb_category_classification.pkl"
DB_PATH        = BASE_DIR /"src" / "DataBase" / "active_learning.db"

# Column sets (mirror train_routes.py)
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


# ─── DB helpers ───────────────────────────────────────────────────────────── #

def _init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS review_queue (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id        TEXT NOT NULL,
            source_name      TEXT,
            binary_prediction  INTEGER,
            binary_confidence  REAL,
            multiclass_prediction TEXT,
            multiclass_confidence REAL,
            uncertainty_score REAL,
            feature_json     TEXT,
            explanation_summary TEXT,
            status           TEXT DEFAULT 'pending',
            queued_at        TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS analyst_reviews (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id            TEXT NOT NULL,
            analyst_label        INTEGER,          -- 0=safe, 1=vuln
            analyst_category     TEXT,
            review_notes         TEXT,
            reviewed_by          TEXT DEFAULT 'analyst',
            reviewed_at          TEXT DEFAULT (datetime('now')),
            source_name          TEXT,
            binary_prediction    INTEGER,
            binary_confidence    REAL,
            multiclass_prediction TEXT,
            multiclass_confidence REAL
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT,
            accuracy   REAL,
            f1         REAL,
            roc_auc    REAL,
            n_train    INTEGER,
            n_analyst  INTEGER,
            recorded_at TEXT DEFAULT (datetime('now'))
        );
        """)


@contextmanager
def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ─── Model loaders ────────────────────────────────────────────────────────── #

def _load_binary():
    if not BINARY_PKL.exists():
        return None, None
    pkg = joblib.load(BINARY_PKL)
    return pkg["model"], pkg["feature_columns"]


def _load_category():
    if not CATEGORY_PKL.exists():
        return None, None, None
    pkg = joblib.load(CATEGORY_PKL)
    return pkg["model"], pkg["feature_columns"], pkg["classes"]


# ─── Uncertainty helpers ───────────────────────────────────────────────────── #

def _uncertainty_score(proba_array: np.ndarray) -> float:
    """
    Least-confidence uncertainty: 1 - P(most_confident_class).
    Range [0, 1]. Higher = more uncertain = more valuable for labelling.
    """
    return float(1.0 - np.max(proba_array))


def _explanation_summary(features: dict, bin_pred: int,
                          bin_proba: float, cat: Optional[str],
                          uncertainty: float) -> str:
    """Generate a one-line analyst-facing explanation."""
    verdict = "VULNERABLE" if bin_pred == 1 else "SAFE"
    conf = round(bin_proba * 100, 1)
    unc  = round(uncertainty * 100, 1)
    hint = f" · Category hint: {cat.upper()}" if cat else ""
    sources = features.get("taint_source_count", 0)
    concat  = features.get("string_concat_near_sink_count", 0)
    dist    = features.get("source_to_sink_line_distance", -1)
    parts = [f"Model says {verdict} ({conf}% confidence, {unc}% uncertainty){hint}."]
    if sources:
        parts.append(f"Found {sources} taint source(s).")
    if concat:
        parts.append(f"{concat} string concat(s) near sink.")
    if dist != -1:
        parts.append(f"Source→sink distance: {dist} lines.")
    return " ".join(parts)


# ─── Routes ───────────────────────────────────────────────────────────────── #

# ── 1. Queue a new scan ───────────────────────────────────────────────────── #

@router.post("/queue")
async def queue_scan(file: UploadFile = File(...)):
    """
    Scan a .java file, compute uncertainty, and add to the review queue.
    Returns the queue entry with uncertainty score.
    """
    if not file.filename.endswith(".java"):
        raise HTTPException(400, "Only .java files accepted.")

    source_code = (await file.read()).decode("utf-8", errors="ignore")
    if not source_code.strip():
        raise HTTPException(400, "File is empty.")

    # Import here to avoid circular at module level
    try:
        import sys, os
        sys.path.append(str(BASE_DIR))
        from src.scanners.java_features import extract_features
        features = extract_features(source_code, file.filename)
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    binary_model, binary_cols = _load_binary()
    cat_model, cat_cols, cat_classes = _load_category()

    if binary_model is None:
        raise HTTPException(503, "Binary model not loaded. Train it first.")

    # Binary prediction
    missing = [c for c in binary_cols if c not in features]
    if missing:
        raise HTTPException(500, f"Missing features: {missing[:5]}…")

    X_bin = pd.DataFrame([{c: features[c] for c in binary_cols}])
    bin_proba_arr = binary_model.predict_proba(X_bin)[0]
    bin_pred = int(np.argmax(bin_proba_arr))
    bin_conf = float(np.max(bin_proba_arr))
    bin_unc  = _uncertainty_score(bin_proba_arr)

    # Category prediction (cascade)
    cat_pred, cat_conf = None, None
    if bin_pred == 1 and cat_model is not None:
        try:
            X_cat = pd.DataFrame([{c: features[c] for c in cat_cols}])
            cat_proba_arr = cat_model.predict_proba(X_cat)[0]
            best_idx = int(np.argmax(cat_proba_arr))
            cat_pred = cat_classes[best_idx]
            cat_conf = float(cat_proba_arr[best_idx])
        except Exception:
            pass

    explanation = _explanation_summary(
        features, bin_pred, bin_conf, cat_pred, bin_unc
    )

    # Sanitize features for JSON (remove non-numeric)
    feat_json = json.dumps({
        k: v for k, v in features.items()
        if isinstance(v, (int, float, bool)) and k != "file_name"
    })

    sample_id = file.filename

    with _db() as con:
        cur = con.execute("""
            INSERT INTO review_queue
              (sample_id, source_name, binary_prediction, binary_confidence,
               multiclass_prediction, multiclass_confidence,
               uncertainty_score, feature_json, explanation_summary)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            sample_id, "upload",
            bin_pred, round(bin_conf, 4),
            cat_pred or "N/A", round(cat_conf, 4) if cat_conf else 0.0,
            round(bin_unc, 4), feat_json, explanation,
        ))
        queue_id = cur.lastrowid

    return JSONResponse({
        "queue_id": queue_id,
        "sample_id": sample_id,
        "binary_prediction": bin_pred,
        "binary_confidence": round(bin_conf, 4),
        "multiclass_prediction": cat_pred or "N/A",
        "multiclass_confidence": round(cat_conf, 4) if cat_conf else 0.0,
        "uncertainty_score": round(bin_unc, 4),
        "explanation_summary": explanation,
        "status": "pending",
    })


# ── 2. Get pending queue ──────────────────────────────────────────────────── #

@router.get("/queue")
def get_queue(limit: int = 50, strategy: str = "uncertainty"):
    """
    Return pending review queue items.
    strategy=uncertainty  → sorted by uncertainty DESC (most uncertain first)
    strategy=confidence   → sorted by confidence DESC (most confident first,
                             useful for verifying easy positives)
    strategy=random       → random order (baseline)
    """
    order = {
        "uncertainty": "uncertainty_score DESC",
        "confidence":  "binary_confidence DESC",
        "random":      "RANDOM()",
    }.get(strategy, "uncertainty_score DESC")

    with _db() as con:
        rows = con.execute(f"""
            SELECT * FROM review_queue
            WHERE status = 'pending'
            ORDER BY {order}
            LIMIT ?
        """, (limit,)).fetchall()

    # Also bucket by strategy tiers for the UI
    pending_count = len(rows)
    high_unc   = [r for r in rows if r["uncertainty_score"] > 0.35]
    boundary   = [r for r in rows if 0.15 < r["uncertainty_score"] <= 0.35]
    multi_flag = [r for r in rows if r["multiclass_prediction"] not in ("N/A", None, "")]

    def row_dict(r):
        d = dict(r)
        d.pop("feature_json", None)   # don't send raw features in list
        return d

    return JSONResponse({
        "total_pending": pending_count,
        "strategy": strategy,
        "high_uncertainty": [row_dict(r) for r in high_unc],
        "boundary_cases":   [row_dict(r) for r in boundary],
        "multi_flagged":    [row_dict(r) for r in multi_flag],
        "all_pending":      [row_dict(r) for r in rows],
    })


# ── 3. Submit an analyst review ───────────────────────────────────────────── #

class ReviewPayload(BaseModel):
    sample_id:            str
    analyst_label:        int              # 0 = safe, 1 = vulnerable
    analyst_category:     Optional[str] = ""
    review_notes:         Optional[str] = ""
    reviewed_by:          Optional[str] = "analyst"
    source_name:          Optional[str] = ""
    binary_prediction:    Optional[int]  = None
    binary_confidence:    Optional[float] = None
    multiclass_prediction:Optional[str]  = None
    multiclass_confidence:Optional[float] = None


@router.post("/reviews")
def submit_review(payload: ReviewPayload):
    """Submit an analyst label for a queued sample."""
    with _db() as con:
        con.execute("""
            INSERT INTO analyst_reviews
              (sample_id, analyst_label, analyst_category, review_notes,
               reviewed_by, source_name,
               binary_prediction, binary_confidence,
               multiclass_prediction, multiclass_confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            payload.sample_id, payload.analyst_label,
            payload.analyst_category or "", payload.review_notes or "",
            payload.reviewed_by or "analyst", payload.source_name or "",
            payload.binary_prediction, payload.binary_confidence,
            payload.multiclass_prediction or "", payload.multiclass_confidence,
        ))
        # Mark queue entry as reviewed
        con.execute("""
            UPDATE review_queue SET status='reviewed'
            WHERE sample_id=? AND status='pending'
        """, (payload.sample_id,))

    return JSONResponse({
        "status": "saved",
        "sample_id": payload.sample_id,
        "analyst_label": payload.analyst_label,
        "analyst_category": payload.analyst_category,
    })


# ── 4. Get completed reviews ──────────────────────────────────────────────── #

@router.get("/reviews")
def get_reviews(limit: int = 200):
    """Return completed analyst reviews with aggregate stats."""
    with _db() as con:
        rows = con.execute("""
            SELECT * FROM analyst_reviews
            ORDER BY reviewed_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        stats = con.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN analyst_label=1 THEN 1 ELSE 0 END) as vuln,
              SUM(CASE WHEN analyst_label=0 THEN 1 ELSE 0 END) as safe_cnt
            FROM analyst_reviews
        """).fetchone()

    return JSONResponse({
        "total_reviews":      stats["total"],
        "vulnerable_reviews": stats["vuln"],
        "safe_reviews":       stats["safe_cnt"],
        "latest_reviews":     [dict(r) for r in rows],
    })


# ── 5. Retrain with analyst labels (active learning loop) ─────────────────── #

@router.post("/retrain-active")
def retrain_active():
    """
    Retrain both models using the original OWASP dataset PLUS any accepted
    analyst reviews.  This closes the active learning loop.

    Strategy:
    - Load original CSV
    - Load analyst_reviews from SQLite
    - For each review, reconstruct a synthetic row: use the queue's
      feature_json as X, analyst_label as y_binary,
      analyst_category (if provided) as y_category
    - Concatenate and retrain (same hyperparams as train_routes.py)
    - Return rich metrics payload showing before/after improvement
    """
    t0 = time.time()

    # ── Load original dataset ────────────────────────────────────────────── #
    if not DATASET_PATH.exists():
        raise HTTPException(404, f"Dataset not found at {DATASET_PATH}")
    df_orig = pd.read_csv(DATASET_PATH)

    # ── Load analyst reviews + their feature vectors ──────────────────────── #
    with _db() as con:
        reviews = con.execute("""
            SELECT ar.*, rq.feature_json
            FROM analyst_reviews ar
            LEFT JOIN review_queue rq
              ON ar.sample_id = rq.sample_id
            WHERE ar.analyst_label IN (0,1)
        """).fetchall()

    analyst_rows = []
    for r in reviews:
        feat_json = r["feature_json"]
        if not feat_json:
            continue
        try:
            feats = json.loads(feat_json)
        except Exception:
            continue
        feats["is_actually_vulnerable"] = int(r["analyst_label"])
        feats["actual_category"]        = r["analyst_category"] or ""
        feats["file_name"]              = r["sample_id"]
        feats["test_name"]              = r["sample_id"]
        feats["actual_cwe"]             = ""
        analyst_rows.append(feats)

    n_analyst = len(analyst_rows)

    if analyst_rows:
        df_analyst = pd.DataFrame(analyst_rows)
        df_combined = pd.concat([df_orig, df_analyst], ignore_index=True, sort=False)
        df_combined = df_combined.fillna(0)
    else:
        df_combined = df_orig.copy()

    # ── Binary retraining ────────────────────────────────────────────────── #
    X_bin = df_combined.drop(columns=[c for c in META_COLS if c in df_combined.columns], errors="ignore")
    y_bin = df_combined["is_actually_vulnerable"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bin, y_bin, test_size=0.20, random_state=42, stratify=y_bin
    )
    bin_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    bin_model.fit(X_tr, y_tr)
    y_pred_bin   = bin_model.predict(X_te)
    y_proba_bin  = bin_model.predict_proba(X_te)[:, 1]
    cm_bin       = confusion_matrix(y_te, y_pred_bin).tolist()

    BINARY_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": bin_model, "feature_columns": list(X_bin.columns)}, BINARY_PKL)

    # ── Category retraining ──────────────────────────────────────────────── #
    df_vuln = df_combined[df_combined["is_actually_vulnerable"] == 1].copy()
    drop_cat = [c for c in LEAKAGE_COLS + META_COLS if c in df_vuln.columns]
    df_vuln_valid = df_vuln[df_vuln["actual_category"].str.strip() != ""]
    X_cat = df_vuln_valid.drop(columns=drop_cat, errors="ignore")
    y_cat = df_vuln_valid["actual_category"]

    cat_metrics = {}
    if len(y_cat.unique()) >= 2:
        X_ctr, X_cte, y_ctr, y_cte = train_test_split(
            X_cat, y_cat, test_size=0.20, random_state=42, stratify=y_cat
        )
        cat_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        )
        cat_model.fit(X_ctr, y_ctr)
        y_pred_cat = cat_model.predict(X_cte)
        classes    = list(cat_model.classes_)
        cm_cat     = confusion_matrix(y_cte, y_pred_cat, labels=classes).tolist()

        CATEGORY_PKL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": cat_model, "feature_columns": list(X_cat.columns),
            "classes": classes, "version": "2.0-active",
            "model_type": "GradientBoostingClassifier",
        }, CATEGORY_PKL)

        cat_metrics = {
            "accuracy":  round(accuracy_score(y_cte, y_pred_cat), 4),
            "macro_f1":  round(f1_score(y_cte, y_pred_cat, average="macro", zero_division=0), 4),
            "confusion_matrix": cm_cat,
            "classes": classes,
        }
    else:
        cat_metrics = {"note": "Not enough categories to retrain — need ≥2."}

    # ── Record metrics ───────────────────────────────────────────────────── #
    with _db() as con:
        con.execute("""
            INSERT INTO model_metrics (model_type, accuracy, f1, roc_auc, n_train, n_analyst)
            VALUES (?,?,?,?,?,?)
        """, (
            "binary",
            round(accuracy_score(y_te, y_pred_bin), 4),
            round(f1_score(y_te, y_pred_bin, zero_division=0), 4),
            round(roc_auc_score(y_te, y_proba_bin), 4),
            len(df_orig), n_analyst,
        ))

    return JSONResponse({
        "retrain_type":    "active_learning",
        "duration_seconds": round(time.time() - t0, 2),
        "dataset": {
            "original_rows": int(len(df_orig)),
            "analyst_rows":  n_analyst,
            "total_rows":    int(len(df_combined)),
        },
        "binary": {
            "accuracy":   round(accuracy_score(y_te, y_pred_bin), 4),
            "roc_auc":    round(roc_auc_score(y_te, y_proba_bin), 4),
            "precision":  round(precision_score(y_te, y_pred_bin, zero_division=0), 4),
            "recall":     round(recall_score(y_te, y_pred_bin, zero_division=0), 4),
            "f1":         round(f1_score(y_te, y_pred_bin, zero_division=0), 4),
            "confusion_matrix": cm_bin,
        },
        "category": cat_metrics,
        "note": f"Models retrained with {n_analyst} analyst label(s) merged in.",
    })


# ── 6. Overview / stats ───────────────────────────────────────────────────── #

@router.get("/overview")
def get_overview():
    """Aggregate platform stats for the dashboard homepage."""
    with _db() as con:
        queue_stats = con.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN status='pending'  THEN 1 ELSE 0 END) as pending,
              SUM(CASE WHEN status='reviewed' THEN 1 ELSE 0 END) as reviewed,
              AVG(uncertainty_score) as avg_uncertainty,
              SUM(CASE WHEN uncertainty_score > 0.35 THEN 1 ELSE 0 END) as high_unc
            FROM review_queue
        """).fetchone()

        rev_stats = con.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN analyst_label=1 THEN 1 ELSE 0 END) as vuln,
              SUM(CASE WHEN analyst_label=0 THEN 1 ELSE 0 END) as safe_cnt
            FROM analyst_reviews
        """).fetchone()

        cat_dist = con.execute("""
            SELECT analyst_category, COUNT(*) as cnt
            FROM analyst_reviews
            WHERE analyst_category != ''
            GROUP BY analyst_category
            ORDER BY cnt DESC
        """).fetchall()

        last_metric = con.execute("""
            SELECT * FROM model_metrics
            ORDER BY recorded_at DESC LIMIT 1
        """).fetchone()

    binary_loaded   = BINARY_PKL.exists()
    category_loaded = CATEGORY_PKL.exists()

    return JSONResponse({
        "models": {
            "binary_loaded":   binary_loaded,
            "category_loaded": category_loaded,
        },
        "queue": {
            "total":          queue_stats["total"]    or 0,
            "pending":        queue_stats["pending"]  or 0,
            "reviewed":       queue_stats["reviewed"] or 0,
            "avg_uncertainty": round(float(queue_stats["avg_uncertainty"] or 0), 3),
            "high_uncertainty": queue_stats["high_unc"] or 0,
        },
        "reviews": {
            "total":     rev_stats["total"]     or 0,
            "vulnerable":rev_stats["vuln"]      or 0,
            "safe":      rev_stats["safe_cnt"]  or 0,
            "category_distribution": {r["analyst_category"]: r["cnt"]
                                       for r in cat_dist},
        },
        "last_retrain": dict(last_metric) if last_metric else None,
    })


# ── 7. Get metric history ──────────────────────────────────────────────────── #

@router.get("/metrics-history")
def get_metrics_history():
    """Return all recorded model metric snapshots (for a learning curve chart)."""
    with _db() as con:
        rows = con.execute("""
            SELECT * FROM model_metrics
            ORDER BY recorded_at ASC
        """).fetchall()
    return JSONResponse({"history": [dict(r) for r in rows]})


# ── Boot: ensure DB exists ─────────────────────────────────────────────────── #
_init_db()
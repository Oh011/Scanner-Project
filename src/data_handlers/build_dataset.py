"""
build_dataset_notebook.py

Same as build_dataset.py, but no command-line arguments needed.
Just edit the CONFIG section below and run the file directly
(e.g. click "Run" in VS Code / PyCharm, or run cells in Jupyter).
"""

import os
import csv
import pandas as pd
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)
from  src.scanners.java_features import extract_features

# ============================================================
# CONFIG -- edit these and just run the file
# ============================================================
TESTCODE_DIR = r"../../data/testcode"
GROUND_TRUTH_CSV = r"../../data/raw/expectedresults-1.2.csv"
OUTPUT_CSV = r"../../data/processed/features_2700_3.csv"
LIMIT = None   # set to None to process every matched file
# ============================================================


def load_ground_truth(path: str) -> dict:
    """
    Returns {test_name: {"category": ..., "cwe": ..., "real_vulnerability": bool}}

    Benchmark's expectedresults-1.2.csv header looks like:
        # test name, category, real vulnerability, cwe, Benchmark version: 1.2, 2016-06-1
    """
    truth = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip().lstrip("#").strip().lower() for h in header]

        idx_name = header.index("test name")
        idx_category = header.index("category")
        idx_vuln = header.index("real vulnerability")
        idx_cwe = header.index("cwe")

        for row in reader:
            if not row or not row[idx_name].strip():
                continue
            test_name = row[idx_name].strip()
            truth[test_name] = {
                "category": row[idx_category].strip(),
                "cwe": row[idx_cwe].strip(),
                "real_vulnerability": row[idx_vuln].strip().lower() == "true",
            }
    return truth


def build_dataset(testcode_dir, ground_truth_csv, output_csv, limit=None):
    ground_truth = load_ground_truth(ground_truth_csv)
    print(f"Loaded {len(ground_truth)} ground truth entries")

    java_files = sorted([f for f in os.listdir(testcode_dir) if f.endswith(".java")])
    print(f"Found {len(java_files)} Java files in {testcode_dir}")

    rows = []
    skipped = 0
    matched_count = 0

    for fname in java_files:
        test_name = fname.replace(".java", "")
        if test_name not in ground_truth:
            skipped += 1
            continue

        if limit is not None and matched_count >= limit:
            break
        matched_count += 1

        filepath = os.path.join(testcode_dir, fname)
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            source_code = f.read()

        features = extract_features(source_code, fname)
        truth = ground_truth[test_name]

        features["test_name"] = test_name
        features["actual_category"] = truth["category"]
        features["actual_cwe"] = truth["cwe"]
        features["is_actually_vulnerable"] = int(truth["real_vulnerability"])

        rows.append(features)

    print(f"Processed {len(rows)} files, skipped {skipped} (no ground truth match)")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved dataset to {output_csv} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print("\nLabel balance:")
    print(df["is_actually_vulnerable"].value_counts())
    print("\nCategory balance:")
    print(df["actual_category"].value_counts())
    return df


if __name__ == "__main__":
    df = build_dataset(TESTCODE_DIR, GROUND_TRUTH_CSV, OUTPUT_CSV, limit=LIMIT)
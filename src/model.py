import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# Paths
# ============================================================

DATASET_PATH = r"../data/processed/features_2700_3.csv"
MODEL_PATH = "models/gb_category_classification.pkl"
OUTPUT_DIR = "models/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)

df_vuln = df[df["is_actually_vulnerable"] == 1].copy()

print(f"Total samples      : {len(df)}")
print(f"Vulnerable samples : {len(df_vuln)}")

# ============================================================
# 2. Remove leakage features
# ============================================================

leakage_cols = [
    "sql_sink_count",
    "cmd_sink_count",
    "xss_sink_count",
    "path_sink_count",
    "ldap_sink_count",
    "xxe_sink_count",
    "weak_crypto_sink_count",

    "xss_sanitizer_count",
    "sql_sanitizer_count",
    "path_sanitizer_count",
    "generic_sanitizer_count",
    "total_sink_count",
    "total_sanitizer_count",
    "sanitizer_to_sink_ratio",
    "has_any_sanitizer",

    "sqli_tainted_sink_reached",
    "sqli_sanitized_before_sink",

    "cmdi_tainted_sink_reached",
    "cmdi_sanitized_before_sink",

    "pathtraver_tainted_sink_reached",
    "pathtraver_sanitized_before_sink",

    "ldapi_tainted_sink_reached",
    "ldapi_sanitized_before_sink",

    "xpathi_tainted_sink_reached",
    "xpathi_sanitized_before_sink"
]

meta_cols = [
    "file_name",
    "test_name",
    "actual_category",
    "actual_cwe",
    "is_actually_vulnerable"
]

X = df_vuln.drop(columns=leakage_cols + meta_cols)
y = df_vuln["actual_category"]

# ============================================================
# 3. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== Dataset Split ==========")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ============================================================
# 4. Train Model
# ============================================================

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# 5. Evaluate
# ============================================================

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n========== Evaluation ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

# Save classification report as CSV
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, "static_multiclass_classification_report.csv"))

# Save confusion matrix as CSV
classes = sorted(y.unique())

cm = confusion_matrix(y_test, predictions, labels=classes)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in classes],
    columns=[f"Predicted {c}" for c in classes]
)
cm_df.to_csv(os.path.join(OUTPUT_DIR, "static_multiclass_confusion_matrix.csv"))

# Save metrics summary
metrics_df = pd.DataFrame([{
    "accuracy": accuracy,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "classes": len(classes)
}])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "static_multiclass_metrics_summary.csv"), index=False)

# ============================================================
# 6. Plots
# ============================================================

# Confusion Matrix
fig, ax = plt.subplots(figsize=(11, 9))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)
disp.plot(values_format="d", ax=ax, xticks_rotation=45)
ax.set_title("Static Multiclass Classifier Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "static_multiclass_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# Per-Class F1 Score
per_class_df = report_df.drop(
    index=["accuracy", "macro avg", "weighted avg"],
    errors="ignore"
).reset_index().rename(columns={"index": "class"})

per_class_df.to_csv(os.path.join(OUTPUT_DIR, "static_multiclass_per_class_metrics.csv"), index=False)

plot_df = per_class_df.sort_values("f1-score", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(plot_df["class"], plot_df["f1-score"])
plt.xlabel("F1-Score")
plt.ylabel("Vulnerability Class")
plt.title("Static Multiclass Per-Class F1 Score")
plt.xlim(0, 1.05)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "static_multiclass_per_class_f1.png"), dpi=300, bbox_inches="tight")
plt.close()

# Feature Importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(os.path.join(OUTPUT_DIR, "static_multiclass_feature_importance.csv"), index=False)

top_features = importance.head(15).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Static Multiclass Classifier Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "static_multiclass_feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 7. Save Model
# ============================================================

model_package = {
    "model": model,
    "feature_columns": list(X.columns),
    "classes": list(model.classes_),
    "version": "1.0",
    "model_type": "GradientBoostingClassifier"
}

joblib.dump(model_package, MODEL_PATH)

print("\n========== Model Saved ==========")
print("Saved as gb_category_classification.pkl")
print("\nSaved files to:", OUTPUT_DIR)
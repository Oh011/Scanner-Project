import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

# ============================================================
# Paths
# ============================================================

DATASET_PATH = r"../data/processed/features_2700_3.csv"
MODEL_PATH = "models/sklearn_gb_model_2.pkl"
OUTPUT_DIR = "models/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================

df = pd.read_csv(DATASET_PATH)

drop_cols = [
    "file_name",
    "test_name",
    "actual_category",
    "actual_cwe",
    "is_actually_vulnerable"
]

X = df.drop(columns=drop_cols)
y = df["is_actually_vulnerable"]

# ============================================================
# 2. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 3. Train Model
# ============================================================

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# 4. Evaluation
# ============================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save classification report as CSV
report = classification_report(
    y_test,
    y_pred,
    target_names=["Safe", "Vulnerable"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, "static_binary_classification_report.csv"))

# Save confusion matrix as CSV
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Safe", "Actual Vulnerable"],
    columns=["Predicted Safe", "Predicted Vulnerable"]
)
cm_df.to_csv(os.path.join(OUTPUT_DIR, "static_binary_confusion_matrix.csv"))

# Save metrics summary
metrics_df = pd.DataFrame([{
    "accuracy": accuracy,
    "roc_auc": roc_auc,
    "test_samples": len(y_test),
    "train_samples": len(y_train)
}])
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "static_binary_metrics_summary.csv"), index=False)

# ============================================================
# 5. Plots
# ============================================================

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Vulnerable"]
)
disp.plot(values_format="d", ax=ax)
ax.set_title("Static Binary Classifier Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "static_binary_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Static Binary Classifier ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "static_binary_roc_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Static Binary Classifier Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "static_binary_precision_recall_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

# Feature Importance
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(os.path.join(OUTPUT_DIR, "static_binary_feature_importance.csv"), index=False)

top_features = importance_df.head(15).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Static Binary Classifier Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "static_binary_feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 6. Save Model
# ============================================================

model_package = {
    "model": model,
    "feature_columns": list(X.columns)
}

joblib.dump(model_package, MODEL_PATH)

print("\nSaved files to:", OUTPUT_DIR)
print("Saved model:", MODEL_PATH)
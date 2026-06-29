import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv(r"../data/processed/features_2700_3.csv")

# ============================================================
# 2. Keep only vulnerable samples
# ============================================================

df_vuln = df[df["is_actually_vulnerable"] == 1].copy()

print(f"Total samples      : {len(df)}")
print(f"Vulnerable samples : {len(df_vuln)}")

# ============================================================
# 3. Remove leakage features
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
# 4. Train/Test Split
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
# 5. Train Model
# ============================================================

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# 6. Evaluate
# ============================================================

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n========== Evaluation ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

# ============================================================
# 7. Save Model
# ============================================================

import pandas as pd

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(importance.head(30))

model_package = {

    # trained classifier
    "model": model,

    # exact feature order used during training
    "feature_columns": list(X.columns),

    # category names
    "classes": list(model.classes_),

    # optional metadata
    "version": "1.0",

    "model_type": "GradientBoostingClassifier"

}

joblib.dump(
    model_package,
    "models/gb_category_classification.pkl"
)

print("\n========== Model Saved ==========")
print("Saved as gb_category_classification.pkl")

print("\nFeatures saved with model:")


for feature in X.columns:
    print(f" - {feature}")
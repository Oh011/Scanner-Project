import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
df = pd.read_csv(r'../data/processed/features_2700_3.csv')

# 2. Preprocessing: Define Features and Target
# Remove metadata/ground truth columns that are not features
drop_cols = ['file_name', 'test_name', 'actual_category', 'actual_cwe', 'is_actually_vulnerable']
X = df.drop(columns=drop_cols)
y = df['is_actually_vulnerable']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train Model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))


model_package = {
    "model": model,
    "feature_columns": list(X.columns)
}

joblib.dump(model_package, "models/sklearn_gb_model_2.pkl")
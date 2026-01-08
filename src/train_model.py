import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_telco_churn.csv")

df = pd.read_csv(DATA_PATH)


X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Evaluation:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))


evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

print("\nDetailed Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

print("\nDetailed Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

import joblib

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "logistic_regression.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(log_reg, MODEL_PATH)

print("Model saved at:", MODEL_PATH)


FEATURES_PATH = os.path.join(BASE_DIR, "..", "model", "feature_names.pkl")
joblib.dump(X.columns.tolist(), FEATURES_PATH)

print("Feature names saved at:", FEATURES_PATH)

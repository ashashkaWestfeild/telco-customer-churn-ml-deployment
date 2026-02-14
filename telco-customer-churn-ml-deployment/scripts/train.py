
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

print("Loading Data...")
# Load Data
df = pd.read_csv("data/processed_telco.csv")

# X and y
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split Data
print("Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
print("Scaling Data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, "model/scaler.pkl")
print("Scaler saved to model/scaler.pkl")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC Score": matthews_corrcoef(y_test, y_pred)
    }
    
    results[name] = metrics
    
    # Save Model
    filename = f"model/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"Saved {name} to {filename}")
    
results_df = pd.DataFrame(results).T
results_df.to_csv("model/metrics.csv")
print("Metrics saved to model/metrics.csv")
print(results_df)

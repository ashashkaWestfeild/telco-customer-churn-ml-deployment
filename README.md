# Telco Customer Churn Prediction using Machine Learning

## 1. Problem Statement
Customer churn is a critical challenge in the telecommunications industry, directly impacting revenue and customer lifetime value.
The objective of this project is to build and compare multiple machine learning classification models to predict whether a customer will churn based on demographic details, service usage, and billing information.

---

## 2. Dataset Description
The dataset used in this project is the **Telco Customer Churn dataset (IBM Sample Data)**.

- **Domain:** Telecommunications
- **Problem Type:** Binary Classification
- **Target Variable:** `Churn`
  - 1 → Customer churned
  - 0 → Customer retained
- **Number of Instances:** ~7,000
- **Number of Features:** 20+ (after preprocessing)

### Key Features:
- Customer demographics (gender, senior citizen, dependents)
- Account information (tenure, contract type, payment method)
- Service usage (internet service, streaming services)
- Billing information (monthly charges, total charges)

The dataset was preprocessed by handling missing values, encoding categorical variables, and preparing it for machine learning models.

---

## 3. Machine Learning Models Used
The following classification models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Random Forest Classifier (Ensemble Model)
6. XGBoost Classifier (Ensemble Model)

---

## 4. Model Evaluation Metrics
Each model was evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.86 | 0.93 | 0.74 | 0.73 | 0.73 | 0.64 |
| Decision Tree | 0.81 | 0.77 | 0.65 | 0.66 | 0.65 | 0.53 |
| KNN | 0.82 | 0.86 | 0.67 | 0.60 | 0.63 | 0.51 |
| Naive Bayes | 0.70 | 0.88 | 0.46 | 0.94 | 0.62 | 0.48 |
| Random Forest | 0.85 | 0.92 | 0.75 | 0.67 | 0.71 | 0.61 |
| XGBoost | 0.86 | 0.93 | 0.75 | 0.72 | 0.73 | 0.64 |

*(Values filled after model evaluation)*

---

## 5. Observations
| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Strong baseline, good balance of precision/recall. |
| Decision Tree | Lower performance, likely overfit despite pruning. |
| KNN | Moderate performance, sensitive to feature scaling. |
| Naive Bayes | High Recall (catches most churners) but very Low Precision (many false alarms). |
| Random Forest | Robust performance, good balance. |
| XGBoost | Best overall performance (tied with LogReg in accuracy but robust). |

These observations highlight the strengths and weaknesses of each model, especially in handling non-linear relationships and class imbalance in churn prediction.

---

## 6. Streamlit Web Application
An interactive **Streamlit web application** was developed to demonstrate the machine learning models.

### Features:
- Upload test dataset (CSV format)
- Select machine learning model
- Display evaluation metrics
- Show confusion matrix / classification report

---

## 7. Project Structure

```text
telco-customer-churn-ml-deployment/
│
├── data/
│ ├── raw_telco.csv
│ └── processed_telco.csv
│
├── model/
│ ├── decision_tree.pkl
│ ├── knn.pkl
│ ├── logistic_regression.pkl
│ ├── metrics.csv
│ ├── naive_bayes.pkl
│ ├── random_forest.pkl
│ ├── scaler.pkl
│ └── xgboost.pkl
│
├── notebooks/
│ ├── all_models_consolidated.ipynb
│ ├── eda.ipynb
│ └── model_training.ipynb
│
├── scripts/
│ └── train.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 8. Deployment
The application is deployed using **Streamlit Community Cloud**.

- **GitHub Repository:** *https://github.com/ashashkaWestfeild/telco-customer-churn-ml-deployment*
- **Live Streamlit App:** *(To be added)*

---

## 9. Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib / Seaborn

---

## 10. Author
**Pritesh Singh**
M.Tech (AIML / DSE)
Machine Learning – Assignment 2

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
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |
| XGBoost | | | | | | |

*(Values will be filled after model evaluation)*

---

## 5. Observations
| ML Model | Observation |
|--------|-------------|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Random Forest | |
| XGBoost | |

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
<<<<<<< HEAD
│ ├── logistic_regression.py
│ ├── decision_tree.py
│ ├── knn.py
│ ├── naive_bayes.py
│ ├── random_forest.py
│ └── xgboost_model.py
│
├── notebooks/
│ └── eda.ipynb
=======
│ ├── scalar.pkl
│ ├── logistic_regression.pkl
│ ├── decision_tree.pkl
│ ├── knn.pkl
│ ├── naive_bayes.pkl
│ ├── random_forest.pkl
│ ├── xgboost.pkl
│ └── metrics.csv
│
├── notebooks/
│ ├── eda.ipynb
│ └── model_training.ipynb
│
├── scripts/
│ └── train.py
>>>>>>> 1dcaa77 (Almost 80%-90% complete)
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


---

## 8. Deployment
The application is deployed using **Streamlit Community Cloud**.

- **GitHub Repository:** *(To be added)*
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

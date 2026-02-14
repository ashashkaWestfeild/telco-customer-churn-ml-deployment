# Telco Customer Churn Prediction - ML Assignment 2

This project demonstrates an end-to-end Machine Learning deployment using Streamlit. It predicts customer churn based on various features.

## Project Structure
- `app.py`: The main Streamlit application file.
- `notebooks/`: Jupyter notebooks for EDA and model training.
- `scripts/`: Python scripts for training models.
- `data/`: Contains the dataset (raw and processed).
- `model/`: Saved trained models and metrics.
- `requirements.txt`: List of dependencies.

## Models Implemented
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (KNN)
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier
6. XGBoost Classifier

## Metrics Evaluated
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## How to Run locally

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Train the models (if not already trained):
    ```bash
    python scripts/train.py
    ```
    OR run the notebook `notebooks/model_training.ipynb`.
4.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Deployment
This app is ready to be deployed on Streamlit Community Cloud.
1.  Push the code to GitHub.
2.  Go to Streamlit Community Cloud.
3.  Connect your GitHub repository.
4.  Select `app.py` as the main file.
5.  Deploy!

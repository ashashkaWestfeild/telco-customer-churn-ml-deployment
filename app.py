
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page config
st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_telco.csv")

df = load_data()

# Load Models & Metrics
@st.cache_resource
def load_models_metrics():
    models = {}
    model_names = ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
    for name in model_names:
        try:
            models[name] = joblib.load(f"model/{name}.pkl")
        except:
            st.error(f"Model {name} not found. Please train models first.")
    
    try:
        metrics = pd.read_csv("model/metrics.csv", index_col=0)
    except:
        metrics = pd.DataFrame()
        st.error("Metrics not found. Please train models first.")
        
    try:
        scaler = joblib.load("model/scaler.pkl")
    except:
        scaler = None
        st.error("Scaler not found. Please train models first.")
        
    return models, metrics, scaler

models, metrics_df, scaler = load_models_metrics()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Model Comparison", "Prediction"])

if page == "EDA":
    st.title("Exploratory Data Analysis")
    
    st.header("Dataset Overview")
    st.write(df.head())
    
    st.header("Statistics")
    st.write(df.describe())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.header("Correlation Heatmap")
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    
    if not metrics_df.empty:
        st.write(metrics_df)
        
        st.header("Metric Visualization")
        metric_to_plot = st.selectbox("Select Metric to Plot", metrics_df.columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=metrics_df.index, y=metrics_df[metric_to_plot], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No metrics available. Please train the models.")

elif page == "Prediction":
    st.title("Customer Churn Prediction")
    
    st.header("Single Customer Prediction")
    
    # ... (Keep existing single prediction logic) ...
    
    input_data = {}
    col1, col2, col3 = st.columns(3)
    
    feature_cols = df.drop('Churn', axis=1).columns.tolist()
    
    for i, col in enumerate(feature_cols):
        with [col1, col2, col3][i % 3]:
            if df[col].nunique() < 10:
                 input_data[col] = st.selectbox(f"Select {col}", sorted(df[col].unique()))
            else:
                 input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))
                 
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    
    if st.button("Predict Single Customer"):
        if scaler and models:
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            model = models[selected_model_name]
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
            
            if prediction == 1:
                st.error(f"Prediction: Churn (Probability: {probability:.2f})" if probability else "Prediction: Churn")
            else:
                st.success(f"Prediction: No Churn (Probability: {probability:.2f})" if probability else "Prediction: No Churn")
        else:
             st.error("Models or Scaler not loaded.")

    st.markdown("---")
    st.header("Batch Prediction (Upload CSV)")

    # Sample CSV for user documentation
    sample_csv = df.drop('Churn', axis=1).head(5).to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Sample CSV for Testing",
        data=sample_csv,
        file_name="sample_telco_churn_test.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload your CSV file for batch prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df_batch = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", input_df_batch.head())

            # Check if all required columns are present
            missing_cols = [col for col in feature_cols if col not in input_df_batch.columns]
            
            if missing_cols:
                st.error(f"Error: The uploaded CSV is missing the following required columns: {missing_cols}")
            else:
                if st.button("Predict Batch"):
                    if scaler and models:
                        # Preprocess and Scale
                        # Ensure columns are in the same order
                        input_batch_processed = input_df_batch[feature_cols]
                        input_batch_scaled = scaler.transform(input_batch_processed)

                        # Predict
                        model = models[selected_model_name]
                        predictions = model.predict(input_batch_scaled)
                        probabilities = model.predict_proba(input_batch_scaled)[:, 1] if hasattr(model, "predict_proba") else [None] * len(predictions)

                        # Add predictions to dataframe
                        results_df = input_df_batch.copy()
                        results_df['Prediction'] = ["Churn" if p == 1 else "No Churn" for p in predictions]
                        if probabilities[0] is not None:
                             results_df['Churn Probability'] = probabilities



                        st.subheader("Batch Prediction Results")
                        st.write(results_df)

                        if 'Churn' in input_df_batch.columns:
                            st.subheader("Batch Evaluation (Ground Truth Available)")
                            y_true = input_df_batch['Churn']
                            y_pred = predictions
                            
                            acc = accuracy_score(y_true, y_pred)
                            st.write(f"**Accuracy on Uploaded Data:** {acc:.4f}")
                            
                            st.write("**Classification Report:**")
                            st.code(classification_report(y_true, y_pred))
                            
                            st.write("**Confusion Matrix:**")
                            cm = confusion_matrix(y_true, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            st.pyplot(fig)

                        # Download results
                        csv_results = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv_results,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )

                    else:
                        st.error("Models or Scaler not loaded.")

        except Exception as e:
            st.error(f"Error processing file: {e}")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page config
st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_telco.csv")

df = load_data()

# Load Metrics
@st.cache_data
def load_metrics():
    if os.path.exists("model/metrics.csv"):
        return pd.read_csv("model/metrics.csv", index_col=0)
    return None

metrics_df = load_metrics()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Model Comparison", "Prediction"])

if page == "EDA":
    st.title("Exploratory Data Analysis")
    
    st.subheader("Dataset Overview")
    st.write(df.head())
    
    st.subheader("Data Statistics")
    st.write(df.describe())
    
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif page == "Model Comparison":
    st.title("Model Comparison")
    
    if metrics_df is not None:
        st.subheader("Model Performance Metrics")
        st.dataframe(metrics_df.style.highlight_max(axis=0))
        
        # Plotting metrics
        st.subheader("Metrics Visualization")
        metric_to_plot = st.selectbox("Select Metric to Plot", metrics_df.columns)
        
        fig, ax = plt.subplots()
        sns.barplot(x=metrics_df.index, y=metrics_df[metric_to_plot], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Metrics file not found. Please train the models first.")

elif page == "Prediction":
    st.title("Customer Churn Prediction")
    
    # Load Scaler
    if os.path.exists("model/scaler.pkl"):
        scaler = joblib.load("model/scaler.pkl")
    else:
        st.error("Scaler not found. Please train the models first.")
        st.stop()
        
    # User Input
    st.subheader("Enter Customer Details")
    
    # Create input fields based on dataframe columns (excluding Churn)
    input_data = {}
    
    # We need to map user friendly names to column names if possible, but for now we use raw names
    # A better approach would be to have the original dataframe to get ranges, but we can infer from processed
    
    # Numerical Columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'numAdminTickets', 'numTechTickets']
    for col in numerical_cols:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))
        
    # Categorical Columns (Encoded) - detailed reconstruction would be complex, 
    # so we will assume simplified input for this assignment or just exposes all features.
    # Given the processed data is OHE, we ideally should have a pipeline. 
    # BUT, based on the assignment request "simple and fast", I will expose the top features or just all features.
    
    # Let's just expose the boolean features as selectboxes
    other_cols = [c for c in df.columns if c not in numerical_cols and c != 'Churn']
    
    for col in other_cols:
        input_data[col] = st.selectbox(col, [0, 1])
        
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Align columns to match training data
    # Ensure column order matches
    input_df = input_df[df.drop('Churn', axis=1).columns]

    
    # Scale Input
    input_scaled = scaler.transform(input_df)
    
    # Select Model
    model_name = st.selectbox("Select Model", [
        "Logistic_Regression", "Decision_Tree", "KNN", 
        "Naive_Bayes", "Random_Forest", "XGBoost"
    ])
    
    # Load Model
    model_path = f"model/{model_name.lower().replace(' ', '_')}.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        if st.button("Predict"):
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            if prediction[0] == 1:
                st.error(f"Prediction: Churn (Probability: {probability[0]:.2f})" if probability is not None else "Prediction: Churn")
            else:
                st.success(f"Prediction: No Churn (Probability: {probability[0]:.2f})" if probability is not None else "Prediction: No Churn")
            
    else:
        st.error(f"Model {model_name} not found. Please train the models first.")


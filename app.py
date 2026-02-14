
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
    
    st.header("Enter Customer Details")
    
    # Create input fields for features
    # Note: We need to match the features used in training.
    # Looking at processed_telco.csv columns would be best, but generic assumption for now based on Telco dataset
    
    # For simplicity in this assignment restoration, we'll assume the user needs to input values 
    # corresponding to the features expected by the model. 
    # Since we used processed_telco.csv, features are already encoded/numerical.
    # A real-world app would take raw inputs and preprocess them. 
    # given the constraints of "restoration", I will inspect X columns from the training script logic 
    # to create inputs.
    # Training script: X = df.drop('Churn', axis=1)
    
    feature_cols = df.drop('Churn', axis=1).columns.tolist()
    
    input_data = {}
    col1, col2, col3 = st.columns(3)
    
    for i, col in enumerate(feature_cols):
        with [col1, col2, col3][i % 3]:
            # Use appropriate input widgets based on data type estimate
            if df[col].nunique() < 10:
                 input_data[col] = st.selectbox(f"Select {col}", sorted(df[col].unique()))
            else:
                 input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))
                 
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    
    if st.button("Predict"):
        if scaler and models:
            # Prepare input
            input_df = pd.DataFrame([input_data])
            
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            model = models[selected_model_name]
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
            
            if prediction == 1:
                st.error(f"Prediction: Churn (Probability: {probability:.2f})" if probability else "Prediction: Churn")
            else:
                st.success(f"Prediction: No Churn (Probability: {probability:.2f})" if probability else "Prediction: No Churn")
        else:
             st.error("Models or Scaler not loaded.")

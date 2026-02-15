
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page config
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4144/4144517.png", width=100) # Placeholder Icon
    st.title("Telco Churn AI")
    st.info("Predict whether a customer will leave likely using advanced Machine Learning.")
    
    st.markdown("---")
    page = st.radio("Navigation", ["🔍 EDA", "📊 Model Comparison", "🔮 Prediction"])
    
    st.markdown("---")
    st.caption("Machine Learning Assignment 2")

# Page Content
if page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Understanding the data distribution and correlations.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.subheader("Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax, palette='viridis')
        st.pyplot(fig)
        
    with col4:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Comparing the performance of different Machine Learning models.")
    
    if not metrics_df.empty:
        # Highlight Best Model
        best_model_name = metrics_df['Accuracy'].idxmax()
        best_accuracy = metrics_df.loc[best_model_name, 'Accuracy']
        
        st.success(f"🏆 Best Performing Model: **{best_model_name}** with **{best_accuracy:.4f}** Accuracy")

        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        st.subheader("Metric Visualization")
        metric_to_plot = st.selectbox("Select Metric to Plot", metrics_df.columns, index=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=metrics_df.index, y=metrics_df[metric_to_plot], ax=ax, palette='magma')
        plt.xticks(rotation=45)
        plt.title(f"Model Comparison - {metric_to_plot}")
        st.pyplot(fig)
    else:
        st.warning("No metrics available. Please train the models.")

elif page == "🔮 Prediction":
    st.title("🔮 Customer Churn Prediction")
    
    tab1, tab2 = st.tabs(["👤 Single Customer", "📂 Batch Prediction (CSV)"])
    
    feature_cols = df.drop('Churn', axis=1).columns.tolist()

    with tab1:
        st.subheader("Single Customer Prediction")
        
        selected_model_name = st.selectbox("Select Model for Prediction", list(models.keys()))
        
        with st.form("prediction_form"):
            input_data = {}
            col1, col2, col3 = st.columns(3)
            
            for i, col in enumerate(feature_cols):
                with [col1, col2, col3][i % 3]:
                    if df[col].nunique() < 10:
                        input_data[col] = st.selectbox(f"{col}", sorted(df[col].unique()))
                    else:
                        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
            
            submitted = st.form_submit_button("Predict Churn Status")
            
            if submitted:
                if scaler and models:
                    input_df = pd.DataFrame([input_data])
                    input_scaled = scaler.transform(input_df)
                    model = models[selected_model_name]
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
                    
                    st.markdown("---")
                    col_res1, col_res2 = st.columns([1, 2])
                    
                    with col_res1:
                        if prediction == 1:
                            st.error("## ⚠️ Churn Predicted")
                        else:
                            st.success("## ✅ Customer Retained")
                    
                    with col_res2:
                        if probability is not None:
                            st.metric("Churn Probability", f"{probability:.2%}")
                            st.progress(probability)
                else:
                    st.error("Models or Scaler not loaded.")
    
    with tab2:
        st.subheader("Batch Prediction (Upload CSV)")
        st.write("Upload a CSV file containing customer data to get predictions for all of them at once.")
        
        # Sample CSV
        sample_csv = df.drop('Churn', axis=1).head(5).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Sample CSV template",
            data=sample_csv,
            file_name="sample_telco_churn_test.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                input_df_batch = pd.read_csv(uploaded_file)
                st.write("### Data Preview")
                st.dataframe(input_df_batch.head())

                # Validation
                missing_cols = [col for col in feature_cols if col not in input_df_batch.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    if st.button("Run Batch Prediction"):
                        with st.spinner("Predicting..."):
                            if scaler and models:
                                input_batch_processed = input_df_batch[feature_cols]
                                input_batch_scaled = scaler.transform(input_batch_processed)

                                model = models[selected_model_name] # Use same model selected in tab 1 or add separate selector
                                predictions = model.predict(input_batch_scaled)
                                probabilities = model.predict_proba(input_batch_scaled)[:, 1] if hasattr(model, "predict_proba") else [None] * len(predictions)

                                results_df = input_df_batch.copy()
                                results_df['Prediction'] = ["Churn" if p == 1 else "No Churn" for p in predictions]
                                if probabilities[0] is not None:
                                     results_df['Churn Probability'] = probabilities

                                st.success("Batch Prediction Complete!")
                                st.dataframe(results_df)
                                
                                # Evaluation if ground truth exists
                                if 'Churn' in input_df_batch.columns:
                                    st.markdown("### 📊 Evaluation against Ground Truth")
                                    y_true = input_df_batch['Churn']
                                    y_pred = predictions
                                    
                                    acc = accuracy_score(y_true, y_pred)
                                    st.metric("Batch Accuracy", f"{acc:.4f}")
                                    
                                    col_eval1, col_eval2 = st.columns(2)
                                    with col_eval1:
                                        st.caption("Confustion Matrix")
                                        cm = confusion_matrix(y_true, y_pred)
                                        fig, ax = plt.subplots()
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                        st.pyplot(fig)
                                    with col_eval2:
                                        st.caption("Classification Report")
                                        st.text(classification_report(y_true, y_pred))

                                csv_results = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="⬇️ Download Results CSV",
                                    data=csv_results,
                                    file_name="churn_predictions.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.error("Models not loaded.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

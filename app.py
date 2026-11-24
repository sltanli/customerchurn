# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "best_churn_model.pkl")
    preprocessor = joblib.load(BASE_DIR / "churn_preprocessor.pkl")
    return model, preprocessor

@st.cache_data
def load_data():
    data = pd.read_csv(
        BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data = data.dropna()

    return data

model, preprocessor = load_model()
df = load_data()

st.title("Customer Churn Prediction Dashboard")

st.success("Machine Learning model loaded successfully!")

st.write("Welcome to our Customer Churn Analysis System.")

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric(
        "Churned Customers",
        (df["Churn"] == "Yes").sum()
    )

with col3:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.2f}%")

with col4:
    st.metric(
        "Average Monthly Charges",
        f"${df['MonthlyCharges'].mean():.2f}"
    )

st.subheader("Dataset Preview")

st.dataframe(df.head(20), use_container_width=True)

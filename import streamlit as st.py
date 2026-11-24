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
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "best_churn_model.pkl")

preprocessor = joblib.load(BASE_DIR / "churn_preprocessor.pkl")

df = pd.read_csv(
    BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

df = df.dropna()

df = df.drop('customerID', axis=1)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")

st.success("Saved ML model loaded successfully!")
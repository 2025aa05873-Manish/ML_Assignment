import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# -------------------------------
# Streamlit App
# -------------------------------
st.title("ML Model Evaluation App 🎯")
st.write("Upload a CSV dataset, choose a model, and view evaluation metrics.")

# 1. Dataset upload
uploaded_file = st.file_uploader("Upload your CSV file (small test data only)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Dataset")
    st.write(data.head())

    # Assume last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 2. Model selection dropdown
    model_options = [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes - Gaussian",
        "Naive Bayes - Multinomial",
        "Random Forest",
    ]
    if xgb_available:
        model_options.append("XGBoost")

    model_choice = st.selectbox("Select a model:", model_options)

    # Model initialization
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    elif model_choice == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
    elif model_choice == "K-Nearest Neighbor Classifier":
        model = KNeighborsClassifier()
    elif model_choice == "Naive Bayes - Gaussian":
        model = GaussianNB()
    elif model_choice == "Naive Bayes - Multinomial":
        model = MultinomialNB()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_choice == "XGBoost" and xgb_available:
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 3. Display evaluation metrics
    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # 4. Confusion matrix and classification report
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
else:
    st.info("Please upload a CSV file to proceed.")

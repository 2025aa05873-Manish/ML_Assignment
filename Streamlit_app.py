import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Import notebooks as modules
import nbimporter

# Assuming you have six notebooks in your GitHub repo:
# logistic_regression.ipynb, decision_tree.ipynb, knn.ipynb,
# naive_bayes_gaussian.ipynb, random_forest.ipynb, xgboost_model.ipynb

from model import logistic_regression
from model import decision_tree
from model import knn
from model import naive_bayes_gaussian
from model import random_forest
from model import xgboost_model

# -------------------------------
# Streamlit App
# -------------------------------
st.title("ML Model Evaluation App 🎯")
st.write("Upload a CSV dataset, choose a model (from GitHub notebooks), and view evaluation metrics.")

# 1. Dataset upload
uploaded_file = st.file_uploader("Upload your CSV file (small test data only)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset Summary")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        st.write(data.head())

        if data.shape[1] < 2:
            st.error("Dataset must have at least one feature column and one target column.")
        else:
            # Split features and target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            # Encode target if categorical
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            # One-hot encode categorical features
            X = pd.get_dummies(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # 2. Model selection dropdown
            model_choice = st.selectbox("Select a model:", [
                "Logistic Regression",
                "Decision Tree Classifier",
                "K-Nearest Neighbor Classifier",
                "Naive Bayes - Gaussian",
                "Random Forest",
                "XGBoost"
            ])

            # Call respective notebook functions
            if model_choice == "Logistic Regression":
                model = logistic_regression.get_model()
            elif model_choice == "Decision Tree Classifier":
                model = decision_tree.get_model()
            elif model_choice == "K-Nearest Neighbor Classifier":
                model = knn.get_model()
            elif model_choice == "Naive Bayes - Gaussian":
                model = naive_bayes_gaussian.get_model()
            elif model_choice == "Random Forest":
                model = random_forest.get_model()
            elif model_choice == "XGBoost":
                model = xgboost_model.get_model()

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

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")


#!/usr/bin/env python
# coding: utf-8

# """
# ================================================================================
# STUDENT INFORMATION
# ================================================================================
# 
# BITS ID: 2025AA05873
# Name: MANISH SINGHAL
# Email: 2025AA05873@WILP.BITS-PILANI.AC.IN
# Date: 14-02-2026
# 
# ===============
# """

# In[1]:


# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import json
import os
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler


# In[2]:


from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
# List top-level contents of your Drive
print(os.listdir("/content/drive/My Drive"))
# Drill down step by step
print(os.listdir("/content/drive/My Drive/"))
print(os.listdir("/content/drive/My Drive/ML_Assignment2_Data"))
# Access files
data_path = '/content/drive/My Drive/ML_Assignment2_Data/'

#df = pd.read_csv(data_path)


# # Task
# Load the 'Credit_card.csv' and 'Credit_card_label.csv' datasets from "/content/drive/My Drive/ML_Assignment2_Data/", preprocess the data, build and train a Logistic Regression model, and then evaluate its performance using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient.

# ## Load Data
# 
# ### Subtask:
# Load 'Credit_card.csv' and 'Credit_card_label.csv' from the specified data path and merge them into a single DataFrame.
# 

# **Reasoning**:
# Load 'Credit_card.csv' and 'Credit_card_label.csv' into DataFrames and then merge them based on a common ID column.
# 
# 

# In[3]:


df_card = pd.read_csv(data_path + 'Credit_card.csv')
df_label = pd.read_csv(data_path + 'Credit_card_label.csv')

# Merge the two dataframes
df_merged = pd.merge(df_card, df_label, on='Ind_ID', how='inner')

print("df_card head:")
print(df_card.head())
print("\ndf_label head:")
print(df_label.head())
print("\nMerged DataFrame head:")
print(df_merged.head())


# In[4]:


display(df_merged.head())


# In[5]:


df_card = pd.read_csv(data_path + 'Credit_card.csv')
df_label = pd.read_csv(data_path + 'Credit_card_label.csv')

# Merge the two dataframes
df_merged = pd.merge(df_card, df_label, on='Ind_ID', how='inner')

print("df_card head:")
print(df_card.head())
print("\ndf_label head:")
print(df_label.head())
print("\nMerged DataFrame head:")
print(df_merged.head())


# ## Data Preprocessing
# 
# ### Subtask:
# Inspect the merged dataset, handle any missing values, encode categorical features, split the data into features (X) and target (y), and then split X and y into training and testing sets. Finally, scale numerical features using StandardScaler.
# 

# 
# To inspect the merged dataset to understand its structure, data types, and identify missing values and categorical features.
# 
# 

# In[6]:


print("DataFrame Info:")
df_merged.info()

print("\nDescriptive Statistics (all columns):")
df_merged.describe(include='all')

print("\nMissing values per column:")
df_merged.isnull().sum()


# 
# Based on the inspection,  missing values are handled by imputing categorical columns (`GENDER`, `Type_Occupation`) with their respective modes and numerical columns (`Annual_income`, `Birthday_count`) with their medians to maintain data integrity and prevent data loss.
# 
# 

# In[7]:


print("Missing values before imputation:")
print(df_merged.isnull().sum()[df_merged.isnull().sum() > 0])

# Impute GENDER with mode
df_merged['GENDER'].fillna(df_merged['GENDER'].mode()[0], inplace=True)

# Impute Annual_income with median
df_merged['Annual_income'].fillna(df_merged['Annual_income'].median(), inplace=True)

# Impute Birthday_count with median
df_merged['Birthday_count'].fillna(df_merged['Birthday_count'].median(), inplace=True)

# Impute Type_Occupation with mode
df_merged['Type_Occupation'].fillna(df_merged['Type_Occupation'].mode()[0], inplace=True)

print("\nMissing values after imputation:")
print(df_merged.isnull().sum()[df_merged.isnull().sum() > 0])


# In[8]:


print("Missing values before imputation:")
print(df_merged.isnull().sum()[df_merged.isnull().sum() > 0])

# Impute GENDER with mode
df_merged['GENDER'] = df_merged['GENDER'].fillna(df_merged['GENDER'].mode()[0])

# Impute Annual_income with median
df_merged['Annual_income'] = df_merged['Annual_income'].fillna(df_merged['Annual_income'].median())

# Impute Birthday_count with median
df_merged['Birthday_count'] = df_merged['Birthday_count'].fillna(df_merged['Birthday_count'].median())

# Impute Type_Occupation with mode
df_merged['Type_Occupation'] = df_merged['Type_Occupation'].fillna(df_merged['Type_Occupation'].mode()[0])

print("\nMissing values after imputation:")
print(df_merged.isnull().sum()[df_merged.isnull().sum() > 0])


# 
# Now that missing values have been handled, I will identify categorical columns and apply one-hot encoding to convert them into numerical format, which is required for most machine learning models. The 'Ind_ID' column will be excluded as it's an identifier, and the 'label' column will be excluded as it's the target variable.
# 
# 

# In[9]:


print("Original DataFrame columns and data types:")
print(df_merged.dtypes)

# Identify categorical columns (excluding 'Ind_ID' and 'label')
categorical_cols = df_merged.select_dtypes(include='object').columns.tolist()

# Exclude 'Ind_ID' as it's an identifier and 'label' as it's the target
if 'Ind_ID' in categorical_cols:
    categorical_cols.remove('Ind_ID')
if 'label' in categorical_cols:
    categorical_cols.remove('label')

print(f"\nCategorical columns to encode: {categorical_cols}")

# Apply one-hot encoding
df_encoded = pd.get_dummies(df_merged, columns=categorical_cols, drop_first=True)

print("\nDataFrame head after one-hot encoding:")
print(df_encoded.head())
print("\nDataFrame info after one-hot encoding:")
df_encoded.info()


# 
# Now that categorical features are encoded, I will separate the DataFrame into features (X) and the target variable (y). The 'label' column is the target, and the 'Ind_ID' column will be dropped from the features as it's an identifier and not relevant for model training.
# 
# 

# In[10]:


X = df_encoded.drop(columns=['label', 'Ind_ID'])
y = df_encoded['label']

print("Features (X) head:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())


#  split the data into training and testing sets using `train_test_split` with a test size of 0.2 and a `random_state` of 42 for reproducibility, as per the subtask instructions.
# 
# 

# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# scale the numerical features in both the training and testing sets using `StandardScaler`. I will fit the scaler only on the training data (`X_train`) to prevent data leakage and then transform both `X_train` and `X_test`.
# 
# 

# In[12]:


from sklearn.preprocessing import StandardScaler

# Identify numerical columns (excluding boolean columns which resulted from one-hot encoding)
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Fit on training data and transform both training and testing data
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("X_train head after scaling numerical features:")
print(X_train.head())
print("\nX_test head after scaling numerical features:")
print(X_test.head())


# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Initialize the Random Forest Classifier
# Using a random_state for reproducibility
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1] # Probability of the positive class

# Calculate evaluation metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_score_rf = roc_auc_score(y_test, y_pred_proba_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
mcc_rf = matthews_corrcoef(y_test, y_pred_rf)

# Print the metrics
print("\n--- Random Forest Classifier Metrics ---")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"AUC Score: {auc_score_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc_rf:.4f}")


#!/usr/bin/env python
# coding: utf-8

# In[8]:


## --- Import libraries ---
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# --- Load datasets ---
train_data = pd.read_csv("train_cleaned.csv")
test_data = pd.read_csv("test_cleaned.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# --- Separate features and target ---
X_train = train_data.drop("Class/ASD", axis=1)
y_train = train_data["Class/ASD"]

# --- Identify numeric and categorical features ---
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# --- One-hot encode categorical features ---
X_train_categorical = pd.get_dummies(X_train[categorical_features])
X_train_numeric = X_train[numeric_features]
X_train_processed = pd.concat([X_train_numeric, X_train_categorical], axis=1)

# --- Process test dataset similarly ---
X_test_numeric = test_data[numeric_features]
X_test_categorical = pd.get_dummies(test_data[categorical_features])
X_test_processed = pd.concat([X_test_numeric, X_test_categorical], axis=1)

# --- Align train and test columns ---
X_test_processed = X_test_processed.reindex(columns=X_train_processed.columns, fill_value=0)

# --- Feature scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# --- Train SVM ---
model = SVC(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# --- Predict on test dataset ---
y_test_pred = model.predict(X_test_scaled)

# --- Prepare Kaggle submission ---
sample_submission["Class/ASD"] = y_test_pred
sample_submission["ID"] = range(1, len(sample_submission) + 1)  # ID starts at 1
sample_submission.to_csv("Prediction_results.csv", index=False)
print("Predictions saved to Prediction_results.csv")


# In[ ]:





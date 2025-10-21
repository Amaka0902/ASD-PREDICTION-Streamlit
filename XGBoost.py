"""
XGBoost ASD Scoring Tool
Trains on 'train_cleaned.csv' and evaluates on 'test_cleaned.csv'
Author: Amaka
"""
"""
ASD XGBoost Model - Train on labeled data, predict on unlabeled test data
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import joblib

# === Load Data ===
train_df = pd.read_csv("train_cleaned.csv")

TARGET_COL = "Class/ASD"

# Convert target to numeric
y_train = train_df[TARGET_COL].replace({'NO': 0, 'YES': 1}).astype(int)
X_train = train_df.drop(columns=[TARGET_COL])

# Detect columns
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# === Preprocessing Pipelines ===
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_cols),
    ("cat", cat_pipeline, cat_cols)
])

# === Train model ===
pos = sum(y_train == 1)
neg = sum(y_train == 0)
scale_pos_weight = neg / pos if pos > 0 else 1.0

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("Training model...")
pipe.fit(X_train, y_train)
print("✅ Training complete.")

# === Predict on training data ===
predictions = pipe.predict(X_train)
probabilities = pipe.predict_proba(X_train)[:, 1]

# === Save predictions ===
output = X_train.copy()
output["ASD_Predicted"] = predictions
output["ASD_Probability"] = probabilities
output.to_csv("asd_xgboostpredictions.csv", index=False)
print("✅ Predictions saved as 'asd_xgboostpredictions.csv'")

# === Save model ===
joblib.dump(pipe, "xgb_asd_scoring_model.joblib")
print("✅ Model saved as 'xgb_asd_scoring_model.joblib'")


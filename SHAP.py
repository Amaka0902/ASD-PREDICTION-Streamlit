import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('train_cleaned.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget variable distribution:")
print(df['Class/ASD'].value_counts())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check unique values in categorical columns
categorical_cols = ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res',
                    'used_app_before', 'age_desc', 'relation']
for col in categorical_cols:
    print(f"\n{col} unique values: {df[col].unique()}")
    print(f"{col} value counts:")
    print(df[col].value_counts())

# Visualize the target variable
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
df['Class/ASD'].value_counts().plot(kind='bar')
plt.title('Target Variable Distribution')
plt.xlabel('Autism Diagnosis')
plt.ylabel('Count')

plt.subplot(2, 3, 2)
df['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')

plt.subplot(2, 3, 3)
df['age'].hist(bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(2, 3, 4)
df['result'].hist(bins=30)
plt.title('Screening Result Distribution')
plt.xlabel('Result Score')
plt.ylabel('Frequency')

plt.subplot(2, 3, 5)
# Calculate total AQ-10 score (sum of A1-A10 scores)
aq_scores = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']].sum(axis=1)
aq_scores.hist(bins=15)
plt.title('AQ-10 Total Score Distribution')
plt.xlabel('AQ-10 Total Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Correlation analysis
plt.figure(figsize=(12, 8))
# Select only numerical columns for correlation
numerical_cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                  'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                  'age', 'result', 'Class/ASD']
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Data preprocessing
# Handle categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Prepare features and target
X = df.drop(['ID', 'Class/ASD', 'age_desc'], axis=1)  # Remove ID, target, and constant column
y = df['Class/ASD']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'result']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Compare model performance
plt.figure(figsize=(8, 6))
models_list = list(results.keys())
accuracy_scores = list(results.values())
plt.bar(models_list, accuracy_scores, color=['skyblue', 'lightcoral'])
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
plt.show()

# Feature importance from Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# Analyze AQ-10 scores by diagnosis
df['AQ_Total'] = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                     'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']].sum(axis=1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Class/ASD', y='AQ_Total', data=df)
plt.title('AQ-10 Total Scores by Autism Diagnosis')
plt.xlabel('Autism Diagnosis (0=No, 1=Yes)')
plt.ylabel('AQ-10 Total Score')

plt.subplot(1, 2, 2)
sns.boxplot(x='Class/ASD', y='result', data=df)
plt.title('Screening Result by Autism Diagnosis')
plt.xlabel('Autism Diagnosis (0=No, 1=Yes)')
plt.ylabel('Screening Result Score')

plt.tight_layout()
plt.show()

# Statistical summary by diagnosis
print("\nAQ-10 Scores by Diagnosis:")
print(df.groupby('Class/ASD')['AQ_Total'].describe())

print("\nScreening Result by Diagnosis:")
print(df.groupby('Class/ASD')['result'].describe())

# Demographic analysis
demographic_features = ['gender', 'ethnicity', 'jaundice', 'austim']
for feature in demographic_features:
    print(f"\n{feature} vs Autism Diagnosis:")
    cross_tab = pd.crosstab(df[feature], df['Class/ASD'], normalize='index') * 100
    print(cross_tab.round(2))

print("\n=== ANALYSIS COMPLETE ===")
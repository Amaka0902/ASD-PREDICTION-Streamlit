import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# === Load trained model and data ===
pipe = joblib.load("xgb_asd_scoring_model.joblib")
train_df = pd.read_csv("train_cleaned.csv")

# === Prepare data ===
TARGET_COL = "Class/ASD"
y_train = train_df[TARGET_COL].replace({'NO': 0, 'YES': 1}).astype(int)
X_train = train_df.drop(columns=[TARGET_COL])

# === Make predictions ===
predictions = pipe.predict(X_train)
probabilities = pipe.predict_proba(X_train)[:, 1]

# === Evaluate metrics ===
accuracy = accuracy_score(y_train, predictions)
precision = precision_score(y_train, predictions)
recall = recall_score(y_train, predictions)
f1 = f1_score(y_train, predictions)
cm = confusion_matrix(y_train, predictions)

# === Print metrics ===
print("\n=== Model Evaluation on Training Data ===")
print(f" Accuracy:  {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_train, predictions, target_names=["Non-ASD", "ASD"]))

# === Save metrics to CSV ===
eval_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [accuracy, precision, recall, f1]
})
eval_results.to_csv("xgb_asd_model_evaluation.csv", index=False)
print("\n Evaluation results saved as 'xgb_asd_model_evaluation.csv'")

# === Visualization 1: Confusion Matrix (Pink Theme) ===
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="pink",
    xticklabels=["Predicted Non-ASD", "Predicted ASD"],
    yticklabels=["Actual Non-ASD", "Actual ASD"]
)
plt.title(" Confusion Matrix - XGBoost Model", fontsize=13, fontweight="bold", color="#C71585")
plt.ylabel("Actual", fontsize=11)
plt.xlabel("Predicted", fontsize=11)
plt.tight_layout()
plt.show()

# === Visualization 2: Bar Chart (Pink Theme) ===
plt.figure(figsize=(6, 4))
sns.barplot(x="Metric", y="Score", data=eval_results, palette="pink")
plt.title(" Model Performance Metrics - XGBoost Classifier", fontsize=13, fontweight="bold", color="#C71585")
plt.ylim(0, 1)
for i, v in enumerate(eval_results["Score"]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10, color="#C71585")
plt.tight_layout()
plt.show()

# === Visualization 3: ROC Curve (Pink Theme) ===
fpr, tpr, thresholds = roc_curve(y_train, probabilities)
roc_auc = roc_auc_score(y_train, probabilities)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#C71585", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

plt.title("ROC Curve - XGBoost ASD Classifier", fontsize=13, fontweight="bold", color="#C71585")
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
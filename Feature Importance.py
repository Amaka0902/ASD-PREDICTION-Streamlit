import xgboost as xgb

xgb_model = model.named_steps["classifier"]  # get XGBoost model inside the pipeline
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title("Top 10 Most Important Features for ASD Prediction")
plt.show()

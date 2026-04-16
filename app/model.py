import joblib
import os
import pandas as pd
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "XGBOOST_FULL.joblib")
MODEL_VERSION = "v1.0.0"

# Load ONCE at startup, not on every request (critical for performance)
_loaded = joblib.load(MODEL_PATH)
model = _loaded['model'] if isinstance(_loaded, dict) else _loaded

# SHAP explainer also loaded once at startup — TreeExplainer is fast for single rows
explainer = shap.TreeExplainer(model)

def predict(features):
    # DataFrame preserves column names — safer than numpy array (order-independent)
    input_df = pd.DataFrame([features.model_dump()])

    # Fraud score (probability of class 1 = fraud)
    fraud_score = float(model.predict_proba(input_df)[0][1])

    # SHAP values for this single transaction
    # Positive value = that feature pushed the score toward fraud
    # Negative value = that feature pushed the score toward legitimate
    shap_values = explainer(input_df)
    shap_scores = {
        feature: round(float(value), 4)
        for feature, value in zip(input_df.columns, shap_values.values[0])
    }

    # Sort by absolute contribution so the biggest driver comes first
    shap_scores = dict(
        sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return fraud_score, shap_scores

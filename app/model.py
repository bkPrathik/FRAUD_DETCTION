import joblib
import os
import math
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

    # Single tree traversal — fraud_score derived from SHAP values directly
    # (avoids a second traversal via predict_proba)
    # log-odds = base_value + sum(shap_values); sigmoid converts to probability
    shap_values = explainer(input_df)
    log_odds = float(shap_values.base_values[0]) + float(shap_values.values[0].sum())
    fraud_score = 1.0 / (1.0 + math.exp(-log_odds))

    shap_scores = {
        feature: round(float(value), 4)
        for feature, value in zip(input_df.columns, shap_values.values[0])
    }
    shap_scores = dict(
        sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return fraud_score, shap_scores

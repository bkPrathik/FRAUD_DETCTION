import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "XGBOOST_FULL.joblib")
MODEL_VERSION = "v1.0.0"

# Load ONCE at startup, not on every request (critical for performance)
_loaded = joblib.load(MODEL_PATH)
model = _loaded['model'] if isinstance(_loaded, dict) else _loaded

def predict(features):
    # DataFrame preserves column names — safer than numpy array (order-independent)
    input_df = pd.DataFrame([features.model_dump()])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    return prediction, probability
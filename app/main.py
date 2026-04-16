from fastapi import FastAPI, HTTPException
from app.input_output import TransactionInput, PredictionOutput
from app import model as model_module
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Returns a fraud score (0-1) for each transaction. "
        "Higher score = higher fraud risk. "
        "Also returns per-feature SHAP contributions: "
        "positive = pushed toward fraud, negative = pushed toward legitimate."
    ),
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """AWS and load balancers will ping this to know if your app is alive."""
    return {"status": "healthy", "model_version": model_module.MODEL_VERSION}

@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    try:
        logger.info(f"Received prediction request: {transaction.model_dump()}")
        fraud_score, shap_scores = model_module.predict(transaction)
        logger.info(f"Fraud score: {fraud_score:.4f} | Top driver: {next(iter(shap_scores))}")
        return PredictionOutput(
            fraud_score=fraud_score,
            model_version=model_module.MODEL_VERSION,
            shap_scores=shap_scores
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

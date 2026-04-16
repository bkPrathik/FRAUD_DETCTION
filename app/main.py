from fastapi import FastAPI, HTTPException
from app.input_output import TransactionInput, PredictionOutput
from app import model as model_module
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Binary classification: 0 = legit, 1 = fraud",
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
        prediction, probability = model_module.predict(transaction)
        logger.info(f"Prediction result: {prediction}, prob: {probability:.4f}")
        return PredictionOutput(
            prediction=prediction,
            fraud_probability=probability,
            model_version=model_module.MODEL_VERSION
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
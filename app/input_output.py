from pydantic import BaseModel, Field
from typing import Dict

class TransactionInput(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_CASH_OUT: float
    type_TRANSFER: float

class PredictionOutput(BaseModel):
    fraud_score: float = Field(
        ...,
        description="Fraud probability between 0 and 1. Higher score = higher fraud risk. Use this to apply your own decisioning threshold."
    )
    model_version: str
    shap_scores: Dict[str, float] = Field(
        ...,
        description="Per-feature contribution to the fraud score. Positive = pushes toward fraud. Negative = pushes toward legitimate. Sorted by absolute contribution, largest first."
    )

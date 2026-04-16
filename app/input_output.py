from pydantic import BaseModel

class TransactionInput(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_CASH_OUT: float
    type_TRANSFER: float

class PredictionOutput(BaseModel):
    prediction: int
    fraud_probability: float
    model_version: str

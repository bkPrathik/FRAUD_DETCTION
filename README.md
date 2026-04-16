# Fraud Detection — XGBoost Scoring API

A real-time fraud scoring service built on XGBoost, served via FastAPI. The model outputs a fraud probability score (0 to 1) for each transaction, making it suitable for rule-based downstream decisioning — flag, block, or review — based on score thresholds.

Fraud types covered: CASH_OUT and TRANSFER transactions.

Trained on the [PaySim synthetic financial dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) — a publicly available simulation of mobile money transactions, widely used as a benchmark for fraud detection modelling. The dataset is not included in this repo; download it from Kaggle if you want to retrain.

---

## What This Is

The model does not return a binary yes/no. It returns a **fraud score** — a probability between 0 and 1. The higher the score, the more likely the transaction is fraudulent. This is intentional: it lets you set your own operational threshold based on your risk appetite (e.g., block anything above 0.9, flag anything above 0.5 for review).

This is standard practice in financial fraud systems. A hard classifier loses information; a score preserves it.

---

## Project Structure

```
├── app/                    
│   ├── main.py             # API endpoints
│   ├── model.py            # Model loading and prediction
│   └── input_output.py     # Request and response schemas
├── notebooks/              
│   ├── 01_data_prep.ipynb
│   ├── 02_xgboost_training.ipynb      # Single full model training
│   ├── 03_xgboost_segments.ipynb      # Segment models (CASH_OUT, TRANSFER)
│   ├── 04_model_comparison.ipynb
│   └── 05_model_test.ipynb
├── training/               
│   └── xgboost_trainer.py  # Optuna hyperparameter tuning wrapper
├── model/                  
│   ├── XGBOOST_FULL.joblib
│   └── XGBOOST_FULL.json
├── requirements.txt
└── README.md
```

---

## Model Training

Two approaches were trained and compared:

**Approach 1 — Single model:** One XGBoost trained on all CASH_OUT and TRANSFER transactions together, with transaction type as binary flags.

**Approach 2 — Segment models:** Two separate XGBoost models, one for CASH_OUT and one for TRANSFER, each trained only on their respective transaction type.

Both approaches used Optuna for hyperparameter tuning with 5-fold stratified cross-validation, optimising for AUC-PR (area under precision-recall curve). KS statistic was tracked as a secondary metric.

**Why AUC-PR over AUC-ROC:** The dataset is heavily imbalanced (fraud is rare). AUC-ROC is misleading in this case because it includes true negatives — a model that never predicts fraud can still score well. AUC-PR focuses only on the fraud class and is a stricter, more meaningful metric here.

---

## Model Comparison

### Training Metrics

| Model | CV AUC-PR | CV KS Stat | Test AUC-PR | Test KS Stat |
|---|---|---|---|---|
| Full model (CASH_OUT + TRANSFER) | 0.9218 | 0.9808 | 0.9248 | 0.9839 |
| Segment — CASH_OUT | 0.6677 | 0.9392 | 0.6527 | 0.9326 |
| Segment — TRANSFER | 0.9952 | 0.9965 | 0.9955 | 0.9965 |

### Score Distribution on Holdout (30% of data, unseen during training)

These tables show how well the model concentrates fraud into the high-score buckets. A good model pushes most fraud cases into the 0.9-1.0 band.

**Full model — all transactions:**

| Score Band | Fraud Cases | Fraud Volume ($) | % of Total Fraud Cases | % of Total Fraud Volume |
|---|---|---|---|---|
| 0.0 - 0.1 | 43 | 5,201,727 | 1.72% | 0.14% |
| 0.1 - 0.2 | 43 | 3,266,912 | 1.72% | 0.09% |
| 0.2 - 0.3 | 21 | 1,622,831 | 0.84% | 0.04% |
| 0.3 - 0.4 | 29 | 3,768,571 | 1.16% | 0.10% |
| 0.4 - 0.5 | 27 | 5,638,568 | 1.08% | 0.16% |
| 0.5 - 0.6 | 39 | 7,411,837 | 1.56% | 0.20% |
| 0.6 - 0.7 | 37 | 5,279,440 | 1.48% | 0.15% |
| 0.7 - 0.8 | 57 | 9,626,188 | 2.28% | 0.26% |
| 0.8 - 0.9 | 99 | 16,324,360 | 3.97% | 0.45% |
| 0.9 - 1.0 | 2101 | 3,577,976,000 | **84.17%** | **98.40%** |

**Segment model — CASH_OUT only:**

| Score Band | Fraud Cases | Fraud Volume ($) | % of Total Fraud Cases | % of Total Fraud Volume |
|---|---|---|---|---|
| 0.0 - 0.1 | 61 | 4,023,539 | 4.88% | 0.22% |
| 0.1 - 0.2 | 68 | 5,897,395 | 5.44% | 0.32% |
| 0.2 - 0.3 | 70 | 8,121,363 | 5.60% | 0.44% |
| 0.3 - 0.4 | 64 | 9,634,420 | 5.12% | 0.52% |
| 0.4 - 0.5 | 66 | 10,167,070 | 5.28% | 0.55% |
| 0.5 - 0.6 | 80 | 15,941,780 | 6.40% | 0.86% |
| 0.6 - 0.7 | 70 | 15,452,930 | 5.60% | 0.83% |
| 0.7 - 0.8 | 91 | 20,960,800 | 7.28% | 1.13% |
| 0.8 - 0.9 | 111 | 34,714,410 | 8.88% | 1.87% |
| 0.9 - 1.0 | 569 | 1,730,464,000 | **45.52%** | **93.27%** |

**Segment model — TRANSFER only:**

| Score Band | Fraud Cases | Fraud Volume ($) | % of Total Fraud Cases | % of Total Fraud Volume |
|---|---|---|---|---|
| 0.0 - 0.1 | 2 | 2,928,374 | 0.16% | 0.17% |
| 0.1 - 0.2 | 0 | 0 | 0.00% | 0.00% |
| 0.2 - 0.3 | 0 | 0 | 0.00% | 0.00% |
| 0.3 - 0.4 | 5 | 398,288 | 0.41% | 0.02% |
| 0.4 - 0.5 | 1 | 10,358 | 0.08% | 0.00% |
| 0.5 - 0.6 | 4 | 337,572 | 0.33% | 0.02% |
| 0.6 - 0.7 | 5 | 24,911 | 0.41% | 0.00% |
| 0.7 - 0.8 | 1 | 10,277 | 0.08% | 0.00% |
| 0.8 - 0.9 | 1 | 20,986 | 0.08% | 0.00% |
| 0.9 - 1.0 | 1200 | 1,708,575,000 | **98.44%** | **99.78%** |

---

## Conclusion: Single Model vs Segment Models

The TRANSFER segment model is the strongest individually — 98.44% of fraud cases and 99.78% of fraud volume land in the top score band, with an AUC-PR of 0.9952. TRANSFER fraud is structurally clean: large amounts, near-zero originating balance after transfer, and the destination account rarely had a prior balance. The signal is strong enough that a dedicated model separates it almost perfectly.

The CASH_OUT segment model is weaker — AUC-PR of 0.6527 and only 45.52% of fraud in the top band. This is not a modelling failure; it reflects the nature of the data. CASH_OUT fraud shares many characteristics with high-value legitimate cash withdrawals. The model struggles to find a clean boundary.

The full single model, despite handling both transaction types together, scores 84.17% of fraud in the top band with a KS of 0.9839. It benefits from a larger training set and learns patterns across both types simultaneously. Crucially, it outperforms the CASH_OUT segment model on every metric.

**The practical choice is the single full model.** It is simpler to deploy (one model, one API call), easier to maintain (one retraining pipeline), and performs better than the CASH_OUT segment model. The TRANSFER segment model does edge it out on that specific segment, but the operational cost of maintaining two separate models with separate holdouts, separate retraining schedules, and routing logic at the API level is not justified by the marginal gain — especially when the deployed model already concentrates 98.40% of fraud volume in the top score band.

If the business later needs to tighten decisioning specifically on TRANSFER transactions, the segment model approach remains a viable upgrade path.

---

## API

### Run locally

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Returns fraud score for a transaction |

### Request

```json
{
  "amount": 261331.82,
  "oldbalanceOrg": 261331.82,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0,
  "type_CASH_OUT": 0.0,
  "type_TRANSFER": 1.0
}
```

### Response

```json
{
  "prediction": 1,
  "fraud_probability": 0.9997,
  "model_version": "v1.0.0"
}
```

Interactive docs (Swagger UI) are available at `http://localhost:8000/docs` when running locally. Once deployed to a server, replace `localhost` with the server's public IP or domain — e.g. `http://54.x.x.x:8000/docs`.

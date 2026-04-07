# Hospital Readmission Prediction System

A production-grade ML engineering project that predicts 30-day hospital readmission risk using public CMS data. The Centers for Medicare & Medicaid Services financially penalizes hospitals whose readmission rates exceed national risk-adjusted expectations — this system scores each hospital's likelihood of incurring those penalties so administrators can act before the measurement period closes.

**Live demo:**
- 🚀 **API docs:** `https://ml-readmission-production.up.railway.app/docs`
- 📊 **Dashboard:** `https://YOUR_USERNAME-ml-readmission-dashboard-app-XXXXX.streamlit.app`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                            │
│                                                                 │
│  data.cms.gov  ──►  ingest.py  ──►  features.py  ──►  train.py │
│  (HRRP API)         parquet         pivot+agg       XGBoost +  │
│                     cache           wide format     MLflow      │
│                                                                 │
│              Orchestrated by Prefect (flow.py)                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ models/model.pkl
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI Service (Railway)                    │
│                                                                  │
│  POST /predict  ──►  model.py  ──►  PredictionResponse          │
│  POST /predict/batch            ──►  BatchPredictionResponse     │
│  GET  /health                                                    │
│  GET  /model/info                                                │
│  GET  /predictions/history  ◄──  monitoring.py (SQLite)         │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HTTP / JSON
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│              Streamlit Dashboard (Community Cloud)               │
│                                                                  │
│  Batch Prediction  │  Model Info  │  Prediction Log             │
└──────────────────────────────────────────────────────────────────┘
```

```
CI/CD
  GitHub push → GitHub Actions → pytest (57 tests) → Docker build check
                                      │
                                   on pass
                                      │
                             Railway auto-deploy
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Data source | CMS HRRP public API | Real business dataset; no licensing issues |
| Pipeline orchestration | Prefect 3 | Task retries, caching, local-first (no server required) |
| Experiment tracking | MLflow | Industry standard; model registry for versioned promotion |
| Hyperparameter optimisation | Optuna | Bayesian TPE search; far more efficient than grid search |
| Model | XGBoost + scikit-learn | Strong tabular baseline; native missing-value handling |
| API | FastAPI + Uvicorn | Async, auto-docs, Pydantic validation out of the box |
| Data validation | Pydantic v2 | Schema enforcement at API boundary; field-level constraints |
| Monitoring | SQLite | Zero-dependency audit log; sufficient for portfolio scale |
| Dashboard | Streamlit | Rapid UI; calls API only — no model code in the frontend |
| Containerisation | Docker (multi-stage) | Slim runtime image; reproducible deploys |
| CI/CD | GitHub Actions | Tests + Docker build on every push and PR |
| Deployment | Railway + Streamlit Cloud | Free tier; production URLs for portfolio |

---

## Local Setup

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Step 5)
- Anaconda or a standard Python installation

### 1. Clone and install

```bash
git clone https://github.com/taehan1024/ml-readmission.git
cd ml-readmission

conda create -n ml-readmission python=3.11 -y
conda activate ml-readmission

pip install -r requirements.txt
```

### 2. Run the data pipeline

```bash
# Download CMS HRRP data (~18k rows, ~2 MB)
python pipeline/ingest.py

# Build feature matrix (one row per hospital, ~3k rows)
python pipeline/features.py

# Train XGBoost with Optuna Bayesian optimization (50 trials, 5-fold CV)
# Start MLflow UI first (optional but recommended):
mlflow ui --port 5000 &

python pipeline/train.py
# open http://localhost:5000 to explore runs

# Control the number of search trials (faster for testing):
python pipeline/train.py --n-trials 20
```

Or run the full pipeline in one command via Prefect:

```bash
python pipeline/flow.py
```

### 3. Start the API

```bash
python -m uvicorn app.main:app --reload --port 8000
# open http://localhost:8000/docs
```

### 4. Start the dashboard

```bash
PYTHONPATH=. streamlit run dashboard/app.py
# open http://localhost:8501
```

### 5. Run with Docker Compose

```bash
# Starts MLflow (port 5000) + FastAPI (port 8000)
docker compose up --build

curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_version":"local"}
```

### 6. Run tests

```bash
pytest tests/ -v
# 57 tests: test_api.py (30) + test_features.py (15) + test_model.py (12)
```

---

## API Reference

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "facility_id": "010001",
    "facility_name": "Example Medical Center",
    "state": "AL",
    "ami_excess_ratio": 1.15,
    "ami_predicted_rate": 14.2,
    "ami_expected_rate": 12.3,
    "ami_discharges": 310,
    "hf_excess_ratio": 1.08,
    "hf_predicted_rate": 23.1,
    "hf_expected_rate": 21.4,
    "hf_discharges": 520,
    "pn_excess_ratio": 0.94,
    "pn_predicted_rate": 16.8,
    "pn_expected_rate": 17.9,
    "pn_discharges": 210
  }'
```

```json
{
  "facility_id": "010001",
  "facility_name": "Example Medical Center",
  "risk_score": 0.7231,
  "risk_level": "high",
  "model_version": "4"
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"ami_excess_ratio": 1.15, "hf_excess_ratio": 1.08},
    {"ami_excess_ratio": 0.88, "hf_excess_ratio": 0.92, "pn_excess_ratio": 0.95}
  ]'
```

```json
{
  "predictions": [
    {"facility_id": null, "facility_name": null, "risk_score": 0.7231, "risk_level": "high", "model_version": "4"},
    {"facility_id": null, "facility_name": null, "risk_score": 0.2847, "risk_level": "low",  "model_version": "4"}
  ],
  "total": 2,
  "high_risk_count": 1,
  "mean_risk_score": 0.5039,
  "model_version": "4"
}
```

### Health check

```bash
curl https://ml-readmission-production.up.railway.app/health
```

```json
{"status": "ok", "model_loaded": true, "model_version": "local"}
```

---

## Model Performance

Evaluated on a held-out 20% validation split (stratified, `random_state=42`). Hyperparameters selected via Optuna TPE Bayesian search over 50 trials with 5-fold stratified cross-validation.

| Metric | Value |
|---|---|
| AUC | 0.9930 |
| F1 Score | 0.9492 |
| Precision | 0.9443 |
| Recall | 0.9542 |
| Brier Score | 0.0349 |

The Brier score (0.035, lower is better, perfect = 0) confirms the model produces well-calibrated probabilities — the scores are meaningful risk estimates, not just ordinal ranks.

---

## Key Findings

**Heart Failure and Pneumonia lead.** `hf_excess_ratio` (12.4% importance) and `pn_excess_ratio` (10.1%) are the strongest predictors. Both are care-coordination-sensitive conditions where post-discharge follow-up quality is the primary lever — hospitals that struggle here tend to have systemic care transition weaknesses, not just disease-specific problems.

**Hip/Knee Replacement is the most diagnostically informative feature.** At 8.4% importance, Hip/Knee Replacement stands out because it is elective surgery on a relatively healthy, predictable patient population. High excess readmissions here cannot be attributed to patient complexity — they signal operational process failures: surgical site infections, poor discharge planning, or inadequate post-acute coordination. A hospital underperforming in Hip/Knee almost certainly has cross-departmental quality issues.

**Importances are distributed across all six conditions.** No single condition overwhelmingly dominates (top feature at 12.4%, not 50%+). High-risk hospitals tend to underperform broadly — penalty risk is a systemic signal, not a condition-specific anomaly.

**"Canary" combination.** The model identifies hospitals elevated on both Heart Failure and Hip/Knee Replacement as near-certain penalty risks. HF elevation alone might be attributed to a complex patient mix; paired Hip/Knee elevation removes that justification and points to organisational care quality.

**Leakage note.** Aggregate excess ratio features (`mean_excess_ratio`, `max_excess_ratio`, `min_excess_ratio`) were excluded from model training because the prediction target is defined as `mean_excess_ratio > 1.0` — including them would constitute target leakage yielding AUC ≈ 1.0 trivially. The individual per-condition excess ratios are retained as legitimate inputs.

---

## Key Technical Decisions

### Why XGBoost over a deep learning model?

The HRRP dataset has ~3,000 rows after pivoting to one-per-hospital. Deep learning requires order-of-magnitude more data for tabular problems. XGBoost handles missing values natively (critical here — CMS suppresses low-volume hospitals), trains in seconds, and produces well-calibrated probabilities. The Brier score on the validation set confirms calibration quality.

### Why Optuna instead of grid search?

A manual grid search over 16 combinations samples a tiny, uniform slice of the hyperparameter space. Optuna's TPE (Tree-structured Parzen Estimator) sampler focuses trials on promising regions identified by earlier results, finding better parameters in fewer evaluations. The 5-fold cross-validation objective also gives a lower-variance estimate of generalisation performance than a single validation split.

### Why pivot long→wide instead of modelling per-measure rows?

The prediction target is *hospital-level penalty risk*, not measure-level risk. Pivoting gives the model cross-measure interaction signals (e.g. "high AMI ratio AND high HF ratio" is a stronger penalty predictor than either alone). The tradeoff is that hospitals with fewer valid measures have more NaN columns — XGBoost handles this without imputation.

### Why SQLite for monitoring instead of a proper time-series store?

For a portfolio project served by a single Railway instance, SQLite has zero operational overhead and no extra cost. The monitoring schema is intentionally simple so it could be replaced with ClickHouse or PostgreSQL without changing the API — `monitoring.py` is the only file that touches the DB.

### Why `MODEL_DOWNLOAD_URL` instead of a Railway volume?

Railway's free tier does not include persistent volumes. Uploading `model.pkl` as a GitHub Release asset (a few MB) and fetching it at container startup is free, versioned alongside the code, and takes ~2 seconds. Updating the model means tagging a new release and redeploying — a deliberate, auditable step.

### Why Streamlit talks to the API and never imports the model directly?

This enforces the correct production pattern: the model lives in one place (the API), and all clients — dashboard, batch jobs, third-party integrations — call the same versioned endpoint. If you update the model, you redeploy the API; the dashboard requires no change.

---

## Project Structure

```
ml-readmission/
├── app/                    FastAPI service
│   ├── main.py             Endpoints and lifespan handler
│   ├── model.py            Model loading (MLflow → download → local)
│   ├── schemas.py          Pydantic request/response models
│   ├── config.py           Pydantic Settings (env vars / .env)
│   └── monitoring.py       SQLite prediction logging
├── pipeline/               ML pipeline
│   ├── ingest.py           CMS HRRP data download + cache
│   ├── features.py         Long→wide pivot + feature engineering
│   ├── train.py            Optuna Bayesian optimization (50 trials, 5-fold CV) + MLflow
│   └── flow.py             Prefect DAG wiring all steps
├── dashboard/              Streamlit frontend
│   ├── app.py              3-tab UI (batch prediction, model info, prediction log)
│   ├── api_client.py       HTTP wrappers for all API calls
│   ├── components.py       Plotly charts + styled tables
│   └── requirements.txt    Streamlit Cloud dependencies
├── tests/                  pytest suite (57 tests)
├── .github/workflows/      GitHub Actions CI
├── Dockerfile              Multi-stage FastAPI image
├── docker-compose.yml      FastAPI + MLflow together
├── railway.toml            Railway deployment config
└── DEPLOY.md               Step-by-step deployment guide
```

---

## Dataset

**CMS Hospital Readmissions Reduction Program (HRRP)**
- Source: [data.cms.gov/provider-data/dataset/9n3s-kdb3](https://data.cms.gov/provider-data/dataset/9n3s-kdb3)
- Coverage: ~3,100 US hospitals, measurement period July 2021 – June 2024
- Measures: AMI, CABG, COPD, Heart Failure, Hip/Knee, Pneumonia (30-day readmissions)
- Public domain — no licence restrictions

---

*Built as an end-to-end ML engineering portfolio project.*

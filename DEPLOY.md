# Deployment Guide

## FastAPI → Railway

### Prerequisites
- Railway account at [railway.app](https://railway.app)
- Railway CLI: `npm install -g @railway/cli` or use the web UI
- Trained `models/model.pkl` on disk (run `python pipeline/train.py` first)

### Step 1 — Upload the model artifact

Railway containers are ephemeral; the model must be fetched at startup.
Upload `models/model.pkl` as a **GitHub Release asset**:

```bash
# Tag the release
git tag v1.0.0
git push origin v1.0.0

# Via GitHub CLI
gh release create v1.0.0 models/model.pkl \
  --title "v1.0.0" \
  --notes "Initial trained model"
```

Copy the asset download URL — it looks like:
`https://github.com/YOUR_USERNAME/ml-readmission/releases/download/v1.0.0/model.pkl`

### Step 2 — Create a Railway project

```bash
railway login
railway init          # creates a new project
railway link          # or link to an existing project
```

Or via the Railway web UI: **New Project → Deploy from GitHub repo**.

### Step 3 — Set environment variables on Railway

In the Railway dashboard → your service → **Variables**, add:

| Variable | Value |
|---|---|
| `MODEL_DOWNLOAD_URL` | GitHub Release asset URL from Step 1 |
| `MODEL_NAME` | `readmission-model` |
| `MODEL_STAGE` | `Production` |
| `LOG_PREDICTIONS` | `true` |
| `MLFLOW_TRACKING_URI` | *(leave blank or omit — MLflow won't be available, API falls back to local pickle)* |

Railway automatically sets `PORT`; `railway.toml` passes it to uvicorn.

### Step 4 — Deploy

```bash
railway up
```

Or push to `main` — Railway auto-deploys on every push if GitHub integration
is enabled.

### Step 5 — Verify

```bash
curl https://YOUR-APP.up.railway.app/health
# {"status":"ok","model_loaded":true,"model_version":"local"}

curl https://YOUR-APP.up.railway.app/docs
# Opens Swagger UI in browser
```

---

## Streamlit Dashboard → Streamlit Community Cloud

### Prerequisites
- GitHub repo is public (or you have a Streamlit Cloud paid plan for private repos)
- FastAPI is deployed and you have its URL

### Step 1 — Push dashboard to GitHub

The dashboard lives at `dashboard/app.py` in the repo root.
Make sure it's committed and pushed.

### Step 2 — Create the app on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo
4. Set **Main file path** to `dashboard/app.py`
5. Click **Advanced settings → Secrets** and paste:

```toml
FASTAPI_URL = "https://YOUR-APP.up.railway.app"
```

6. Click **Deploy**

### Step 3 — Verify

The dashboard URL will be:
`https://YOUR_USERNAME-ml-readmission-dashboard-app-XXXXX.streamlit.app`

Open it and confirm the sidebar shows "🟢 Connected".

---

## Local end-to-end smoke test (before deploying)

```bash
# 1. Train model
python pipeline/train.py --no-registry

# 2. Start API
uvicorn app.main:app --port 8000 &

# 3. Start dashboard
streamlit run dashboard/app.py &

# 4. Quick API check
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ami_excess_ratio": 1.15, "hf_excess_ratio": 0.95, "pn_excess_ratio": 1.08}'
```

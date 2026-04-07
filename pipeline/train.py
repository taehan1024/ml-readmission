"""pipeline/train.py

Train an XGBoost readmission-risk classifier with MLflow experiment tracking.

Steps
-----
1. Load feature matrix from ``data/processed/features.parquet``
2. Split into train (80%) / validation (20%) sets (stratified)
3. Run Optuna Bayesian hyperparameter optimisation with 5-fold stratified CV
   on the training set; log every trial to MLflow as a nested run
4. Retrain final model on full training set with best params
5. Promote best model to the MLflow Model Registry as "Production"
6. Save best model locally to ``models/model.pkl`` as a fallback
7. Print final AUC and classification report on the held-out validation set

Usage
-----
    python pipeline/train.py
    python pipeline/train.py --features path/to/features.parquet
    python pipeline/train.py --n-trials 30
    python pipeline/train.py --no-registry   # skip MLflow registry promotion
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

EXPERIMENT_NAME = "readmission-prediction"
REGISTERED_MODEL_NAME = "readmission-model"
TARGET_COL = "high_readmission_risk"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
DEFAULT_N_TRIALS = 50

# Columns to drop before training (identifiers, raw state string, and aggregate
# excess-ratio features that directly encode the target).
#
# DATA LEAKAGE NOTE: the target is defined as mean_excess_ratio > 1.0, so
# mean_excess_ratio, max_excess_ratio, min_excess_ratio, and
# n_measures_over_threshold are all trivially predictive of the label.
# Removing them forces the model to generalise from per-condition metrics.
DROP_COLS = [
    "facility_id", "facility_name", "state", TARGET_COL,
    "mean_excess_ratio", "max_excess_ratio", "min_excess_ratio",
    "n_measures_over_threshold",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and separate X from y.

    Parameters
    ----------
    path:
        Path to ``data/processed/features.parquet``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix X and binary target series y.

    Raises
    ------
    FileNotFoundError
        If the features parquet does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. "
            "Run `python pipeline/features.py` first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded features: %d rows x %d cols", *df.shape)

    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS, errors="ignore")

    logger.info(
        "Target distribution — high-risk: %d (%.1f%%), low-risk: %d (%.1f%%)",
        y.sum(), 100 * y.mean(),
        (y == 0).sum(), 100 * (1 - y.mean()),
    )
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    model: XGBClassifier, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Compute classification metrics for a fitted model on a held-out set.

    Parameters
    ----------
    model:
        A fitted XGBClassifier.
    X:
        Feature matrix (held-out validation set).
    y:
        True binary labels.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: auc, f1, precision, recall, brier_score.
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, proba),
        "f1": f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "brier_score": brier_score_loss(y, proba),
    }


# ── Optuna Bayesian optimisation with 5-fold CV ───────────────────────────────

def optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = DEFAULT_N_TRIALS,
) -> tuple[XGBClassifier, dict[str, float], dict[str, Any]]:
    """Bayesian hyperparameter search via Optuna with 5-fold stratified CV.

    Each trial samples hyperparameters using the TPE sampler, evaluates them
    via 5-fold stratified cross-validation on X_train, and logs the trial as
    a child MLflow run. After all trials, the best parameters are used to
    retrain a final model on the full X_train; that model is evaluated once on
    the held-out X_val and returned.

    Parameters
    ----------
    X_train, y_train:
        Training split (CV is performed entirely within this set).
    X_val, y_val:
        Held-out validation split for final evaluation only.
    n_trials:
        Number of Optuna trials (default 50).

    Returns
    -------
    tuple[XGBClassifier, dict, dict]
        (best_model, val_metrics, best_params)
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": round(scale_pos_weight, 4),
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
        }

        fold_aucs: list[float] = []
        for fold_X_tr, fold_X_vl, fold_y_tr, fold_y_vl in (
            (X_train.iloc[tr], X_train.iloc[vl], y_train.iloc[tr], y_train.iloc[vl])
            for tr, vl in cv.split(X_train, y_train)
        ):
            m = XGBClassifier(**params)
            m.fit(fold_X_tr, fold_y_tr, verbose=False)
            fold_aucs.append(roc_auc_score(fold_y_vl, m.predict_proba(fold_X_vl)[:, 1]))

        cv_auc_mean = float(np.mean(fold_aucs))
        cv_auc_std = float(np.std(fold_aucs))

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.log_params(params)
            mlflow.log_metric("cv_auc_mean", cv_auc_mean)
            mlflow.log_metric("cv_auc_std", cv_auc_std)

        logger.info(
            "Trial %d: cv_auc=%.4f ± %.4f",
            trial.number, cv_auc_mean, cv_auc_std,
        )
        return cv_auc_mean

    logger.info("Starting Optuna search: %d trials, %d-fold CV", n_trials, N_CV_FOLDS)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    logger.info(
        "Best trial %d: cv_auc=%.4f  params=%s",
        best_trial.number, best_trial.value,
        {k: v for k, v in best_trial.params.items()
         if k not in ("scale_pos_weight", "eval_metric", "random_state", "tree_method")},
    )

    # Retrain final model on full training set with best params
    best_params_full: dict[str, Any] = {
        **best_trial.params,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
    }
    best_model = XGBClassifier(**best_params_full)
    best_model.fit(X_train, y_train, verbose=False)

    val_metrics = compute_metrics(best_model, X_val, y_val)
    # Also record the CV AUC alongside val metrics
    val_metrics["cv_auc_mean"] = round(best_trial.value, 6)

    return best_model, val_metrics, best_params_full


# ── Feature importances ───────────────────────────────────────────────────────

def log_feature_importances(
    model: XGBClassifier, feature_names: list[str]
) -> None:
    """Log feature importances as MLflow params and print top-10 to stdout.

    Parameters
    ----------
    model:
        Fitted XGBClassifier with ``feature_importances_`` attribute.
    feature_names:
        Ordered list of feature column names matching training columns.
    """
    importances = model.feature_importances_
    fi_pairs = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    # Log top-20 to MLflow
    for rank, (name, score) in enumerate(fi_pairs[:20], 1):
        mlflow.log_metric(f"fi_{rank:02d}_{name}", round(float(score), 6))

    print("\n  Top-10 Feature Importances:")
    for name, score in fi_pairs[:10]:
        bar = "#" * int(score * 200)
        print(f"    {name:<45}  {score:.4f}  {bar}")


# ── Model registry ────────────────────────────────────────────────────────────

def register_model(run_id: str, use_registry: bool) -> str | None:
    """Register the best model in the MLflow Model Registry as 'Production'.

    Parameters
    ----------
    run_id:
        MLflow run ID of the parent (best-model) run.
    use_registry:
        When False, skip registration (useful if no registry backend).

    Returns
    -------
    str | None
        The model version string if registered, else None.
    """
    if not use_registry:
        logger.info("Skipping MLflow model registry (--no-registry flag set).")
        return None

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    try:
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
        version = mv.version
        logger.info(
            "Registered model '%s' version %s", REGISTERED_MODEL_NAME, version
        )

        # Transition to Production (legacy stages API — works with local MLflow)
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Model version %s transitioned to Production stage.", version
        )
        return str(version)

    except Exception as exc:
        logger.warning(
            "Model registry step failed (%s). "
            "Model is still saved in the run artifacts.",
            exc,
        )
        return None


# ── Local fallback save ───────────────────────────────────────────────────────

def save_local(model: XGBClassifier, feature_names: list[str]) -> Path:
    """Pickle the best model to ``models/model.pkl`` as a local fallback.

    Saves a dict containing the model and its expected feature names so
    the API can validate incoming requests at inference time.

    Parameters
    ----------
    model:
        Fitted XGBClassifier to persist.
    feature_names:
        Ordered list of feature column names.

    Returns
    -------
    Path
        Path to the saved pickle file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "model.pkl"
    payload = {
        "model": model,
        "feature_names": feature_names,
        "model_name": REGISTERED_MODEL_NAME,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Local fallback model saved to %s", out_path)
    return out_path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def train(
    features_path: Path = DEFAULT_FEATURES_PATH,
    use_registry: bool = True,
    n_trials: int = DEFAULT_N_TRIALS,
) -> XGBClassifier:
    """Run the full training pipeline.

    Parameters
    ----------
    features_path:
        Path to the processed feature parquet.
    use_registry:
        Whether to promote the best model to the MLflow registry.
    n_trials:
        Number of Optuna hyperparameter search trials.

    Returns
    -------
    XGBClassifier
        The best fitted model.
    """
    X, y = load_features(features_path)
    feature_names = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Train: %d rows | Val: %d rows", len(X_train), len(X_val)
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="best_model") as parent_run:
        # Log dataset metadata
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_cv_folds", N_CV_FOLDS)
        mlflow.log_param("target_pos_rate", round(float(y_train.mean()), 4))

        best_model, best_metrics, best_params = optuna_search(
            X_train, y_train, X_val, y_val, n_trials=n_trials
        )

        # Log best params and metrics to the parent run
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        log_feature_importances(best_model, feature_names)

        # Log model artifact
        mlflow.xgboost.log_model(best_model, artifact_path="model")

        run_id = parent_run.info.run_id
        logger.info("Parent MLflow run ID: %s", run_id)

    # Register best model (outside the run context to avoid nesting issues)
    register_model(run_id, use_registry)

    # Local fallback
    save_local(best_model, feature_names)

    _report(best_model, best_metrics, X_val, y_val)
    return best_model


# ── Report ────────────────────────────────────────────────────────────────────

def _report(
    model: XGBClassifier,
    metrics: dict[str, float],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Print final metrics and classification report to stdout.

    Parameters
    ----------
    model:
        Fitted model used for prediction.
    metrics:
        Pre-computed metric dict from :func:`compute_metrics`.
    X_val:
        Validation feature matrix.
    y_val:
        True validation labels.
    """
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print(f"\n{'='*60}")
    print("  Training Complete — Best Model Metrics")
    print(f"{'='*60}")
    print(f"  AUC         : {metrics['auc']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  Brier Score : {metrics['brier_score']:.4f}")
    print(f"\n{classification_report(y_val, preds, target_names=['Low Risk', 'High Risk'])}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run model training."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost readmission model with MLflow tracking."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to feature parquet (default: data/processed/features.parquet)",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Skip MLflow model registry promotion.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of Optuna search trials (default: {DEFAULT_N_TRIALS}).",
    )
    args = parser.parse_args()

    try:
        train(
            features_path=args.features,
            use_registry=not args.no_registry,
            n_trials=args.n_trials,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Training failed unexpectedly")
        sys.exit(1)


if __name__ == "__main__":
    main()

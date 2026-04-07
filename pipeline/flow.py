"""pipeline/flow.py

Prefect flow that wires together the full ML pipeline:
    ingest → features → train

Each step is a Prefect @task so failures are reported individually and
retries are handled automatically. The enclosing @flow provides a single
entry point and produces a run record in the Prefect UI (or local logs
when no Prefect server is running).

Usage
-----
    python pipeline/flow.py                          # full pipeline, default paths
    python pipeline/flow.py --force-ingest           # re-download raw data
    python pipeline/flow.py --no-registry            # skip MLflow model registry
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

import pandas as pd

from pipeline.ingest import download_hrrp
from pipeline.features import build_features
from pipeline.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "hrrp_raw.parquet"
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(
    name="ingest-cms-data",
    description="Download CMS HRRP dataset and cache as parquet.",
    retries=2,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
)
def ingest_task(force: bool = False) -> Path:
    """Download CMS HRRP data and return the path to the cached parquet.

    Parameters
    ----------
    force:
        Re-download even if a local cache exists.

    Returns
    -------
    Path
        Path to ``data/raw/hrrp_raw.parquet``.
    """
    logger = get_run_logger()
    logger.info("Starting ingest (force=%s)", force)
    download_hrrp(force=force)
    logger.info("Ingest complete: %s", DEFAULT_RAW_PATH)
    return DEFAULT_RAW_PATH


@task(
    name="build-features",
    description="Pivot CMS long-format data and engineer hospital-level features.",
    retries=1,
    retry_delay_seconds=5,
)
def features_task(raw_path: Path) -> Path:
    """Run feature engineering and return the path to the processed parquet.

    Parameters
    ----------
    raw_path:
        Path to the raw HRRP parquet produced by :func:`ingest_task`.

    Returns
    -------
    Path
        Path to ``data/processed/features.parquet``.
    """
    logger = get_run_logger()
    logger.info("Building features from %s", raw_path)
    build_features(raw_path=raw_path, out_path=DEFAULT_FEATURES_PATH)
    logger.info("Features complete: %s", DEFAULT_FEATURES_PATH)
    return DEFAULT_FEATURES_PATH


@task(
    name="train-model",
    description="Train XGBoost classifier with MLflow experiment tracking.",
    retries=1,
    retry_delay_seconds=5,
)
def train_task(features_path: Path, use_registry: bool = True) -> str:
    """Train the readmission model and return the local model path.

    Parameters
    ----------
    features_path:
        Path to the processed feature parquet from :func:`features_task`.
    use_registry:
        Whether to register the best model in the MLflow Model Registry.

    Returns
    -------
    str
        Absolute path to the saved ``models/model.pkl`` fallback file.
    """
    logger = get_run_logger()
    logger.info("Starting model training (registry=%s)", use_registry)
    train(features_path=features_path, use_registry=use_registry)
    model_path = str(PROJECT_ROOT / "models" / "model.pkl")
    logger.info("Training complete. Model saved to %s", model_path)
    return model_path


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(
    name="readmission-prediction-pipeline",
    description=(
        "End-to-end pipeline: download CMS HRRP data → engineer features → "
        "train XGBoost readmission model with MLflow tracking."
    ),
    log_prints=True,
)
def readmission_pipeline(
    force_ingest: bool = False,
    use_registry: bool = True,
) -> str:
    """Run the full readmission prediction pipeline.

    Executes three steps sequentially:
    1. **Ingest** — download CMS HRRP data (cached by default).
    2. **Features** — pivot and engineer hospital-level features.
    3. **Train** — fit XGBoost with hyperparameter grid search, log to MLflow.

    Parameters
    ----------
    force_ingest:
        Pass ``True`` to re-download raw CMS data even if a cache exists.
    use_registry:
        Pass ``False`` to skip promoting the model to the MLflow registry.

    Returns
    -------
    str
        Path to the saved model pickle (``models/model.pkl``).
    """
    raw_path = ingest_task(force=force_ingest)
    features_path = features_task(raw_path)
    model_path = train_task(features_path, use_registry=use_registry)
    return model_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the Prefect flow."""
    parser = argparse.ArgumentParser(
        description="Run the readmission prediction pipeline as a Prefect flow."
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Re-download CMS data even if local cache exists.",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Skip MLflow model registry promotion.",
    )
    args = parser.parse_args()

    result = readmission_pipeline(
        force_ingest=args.force_ingest,
        use_registry=not args.no_registry,
    )
    print(f"\nPipeline complete. Model artifact: {result}")


if __name__ == "__main__":
    main()

"""app/config.py

Application configuration via environment variables.

All settings can be overridden by a ``.env`` file in the project root or
by setting environment variables directly. See ``.env.example`` for a
complete list with descriptions.

Usage
-----
    from app.config import settings
    print(settings.mlflow_tracking_uri)
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file.

    Attributes
    ----------
    mlflow_tracking_uri:
        URI of the MLflow tracking server.
    model_name:
        Registered model name in the MLflow Model Registry.
    model_stage:
        Registry stage to load (``Production``, ``Staging``, or ``None``).
    model_local_path:
        Fallback path for the pickled model, relative to project root.
    log_predictions:
        Whether to persist every prediction to SQLite.
    monitoring_db_path:
        Path to the SQLite monitoring database.
    api_host:
        Host for uvicorn (used by Docker / Railway).
    api_port:
        Port for uvicorn.
    """

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "readmission-model"
    model_stage: str = "Production"
    model_local_path: str = "models/model.pkl"
    # Optional HTTPS URL to download model.pkl at container startup.
    # Set this on Railway so the container fetches the artifact without
    # needing a persistent volume (e.g. a GitHub Release asset URL).
    model_download_url: str = ""
    log_predictions: bool = True
    monitoring_db_path: str = "data/predictions.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def model_local_abs_path(self) -> Path:
        """Absolute path to the local fallback model pickle."""
        p = Path(self.model_local_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def monitoring_db_abs_path(self) -> Path:
        """Absolute path to the SQLite monitoring database."""
        p = Path(self.monitoring_db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Returns
    -------
    Settings
        Application settings loaded once from env / .env file.
    """
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()

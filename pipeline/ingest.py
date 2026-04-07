"""pipeline/ingest.py

Download CMS Hospital Readmissions Reduction Program (HRRP) data from
data.cms.gov and cache locally as a parquet file.

Dataset: https://data.cms.gov/provider-data/dataset/9n3s-kdb3
Measures 30-day excess readmission ratios for ~3,000 US hospitals across
six conditions (AMI, CABG, COPD, Heart Failure, Hip/Knee, Pneumonia).

Usage
-----
    python pipeline/ingest.py                      # download only if not cached
    python pipeline/ingest.py --force              # re-download unconditionally
    python pipeline/ingest.py --max-rows 3000      # trial run — stop after N rows
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Root of the project (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "hrrp_raw.parquet"
PARTIAL_PATH = RAW_DIR / "hrrp_raw_partial.parquet"
CHECKPOINT_PATH = RAW_DIR / ".ingest_checkpoint.json"

# CMS Provider Data Catalog — datastore query API
# resource_id: 9n3s-kdb3  (Hospital Readmissions Reduction Program)
CMS_API_URL = (
    "https://data.cms.gov/provider-data/api/1/datastore/query/9n3s-kdb3/0"
)
PAGE_SIZE = 500            # rows per API request (smaller = more reliable)
REQUEST_TIMEOUT = 120      # seconds per request
MAX_RETRIES = 5
RETRY_BACKOFF = 5.0        # seconds between retries
PAGE_SLEEP = 1.0           # seconds between successful page fetches
BACKOFF_503 = 30.0         # extra sleep on 503 Service Unavailable


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    """Load checkpoint state from disk, or return defaults if missing."""
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            pass
    return {"offset": 0, "total": 0}


def _save_checkpoint(offset: int, total: int) -> None:
    """Atomically write checkpoint state to disk."""
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"offset": offset, "total": total}))
    tmp.replace(CHECKPOINT_PATH)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_page(offset: int, session: requests.Session) -> dict:
    """Fetch one page of results from the CMS datastore query API.

    Parameters
    ----------
    offset:
        Row offset (0-indexed) for pagination.
    session:
        Persistent requests.Session for connection reuse.

    Returns
    -------
    dict
        Parsed JSON response containing ``results`` and ``count`` keys.

    Raises
    ------
    requests.HTTPError
        If the server returns a non-2xx status after all retries.
    """
    params = {
        "limit": PAGE_SIZE,
        "offset": offset,
        "count": "true",
        "results": "true",
    }
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(CMS_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            # Give the server a longer break on 503 before raising
            if resp.status_code == 503 and attempt < MAX_RETRIES:
                logger.warning(
                    "503 Service Unavailable (attempt %d/%d) — backing off %.0fs",
                    attempt, MAX_RETRIES, BACKOFF_503,
                )
                time.sleep(BACKOFF_503)
                last_exc = requests.HTTPError(response=resp)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                logger.warning(
                    "Request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise RuntimeError(
        f"Failed to fetch offset={offset} after {MAX_RETRIES} attempts"
    ) from last_exc


def _append_partial(batch: list[dict]) -> None:
    """Append a batch of rows to the partial parquet file on disk."""
    new_df = pd.DataFrame(batch)
    if PARTIAL_PATH.exists():
        existing = pd.read_parquet(PARTIAL_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(PARTIAL_PATH, index=False)


def download_hrrp(force: bool = False, max_rows: int | None = None) -> pd.DataFrame:
    """Download HRRP data from CMS and return as a DataFrame.

    Paginates through the full dataset (or up to ``max_rows``) and saves
    progress incrementally so a failed run can be resumed.
    On success, writes ``data/raw/hrrp_raw.parquet``.

    Parameters
    ----------
    force:
        When True, re-download even if the local cache already exists.
    max_rows:
        Stop after fetching this many rows (useful for trial runs).
        ``None`` means fetch the full dataset.

    Returns
    -------
    pd.DataFrame
        Raw HRRP data with original CMS column names.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        logger.info(
            "Cache found at %s — loading (use --force to re-download)", OUTPUT_PATH
        )
        df = pd.read_parquet(OUTPUT_PATH)
        _report(df)
        return df

    # --force: wipe any partial state and start fresh
    if force:
        for p in (PARTIAL_PATH, CHECKPOINT_PATH):
            if p.exists():
                p.unlink()

    # Resume from checkpoint if available
    checkpoint = _load_checkpoint()
    resume_offset: int = checkpoint["offset"]
    total_rows: int = checkpoint["total"]

    if resume_offset > 0 and PARTIAL_PATH.exists():
        logger.info("Resuming download from offset %d (of %d total rows) ...", resume_offset, total_rows)
    else:
        logger.info("Starting download from CMS datastore API ...")
        resume_offset = 0

    with requests.Session() as session:
        session.headers.update({"Accept": "application/json"})

        if resume_offset == 0:
            # First request to discover total row count
            first_page = _get_page(offset=0, session=session)
            total_rows = first_page.get("count", 0)
            if max_rows is not None:
                total_rows = min(total_rows, max_rows)
            logger.info("Total rows to fetch: %d", total_rows)
            _append_partial(first_page["results"])
            _save_checkpoint(offset=PAGE_SIZE, total=total_rows)
            offset = PAGE_SIZE
        else:
            offset = resume_offset
            if max_rows is not None:
                total_rows = min(total_rows, max_rows)

        while offset < total_rows:
            page = _get_page(offset=offset, session=session)
            batch = page.get("results", [])
            if not batch:
                break
            _append_partial(batch)
            offset += PAGE_SIZE
            _save_checkpoint(offset=offset, total=total_rows)
            fetched = min(offset, total_rows)
            logger.info("  fetched %d / %d rows ...", fetched, total_rows)
            time.sleep(PAGE_SLEEP)

    # Finalise: promote partial → output, clean up checkpoint
    df = pd.read_parquet(PARTIAL_PATH)
    df.to_parquet(OUTPUT_PATH, index=False)
    PARTIAL_PATH.unlink()
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    logger.info("Download complete — %d rows, %d columns", len(df), df.shape[1])
    logger.info("Saved to %s", OUTPUT_PATH)

    _report(df)
    return df


def _report(df: pd.DataFrame) -> None:
    """Print a summary of the downloaded dataset to stdout.

    Parameters
    ----------
    df:
        The loaded HRRP DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"  HRRP Dataset Summary")
    print(f"{'='*60}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"\n  Column names:")
    for col in df.columns:
        print(f"    - {col}")
    print(f"{'='*60}\n")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the download."""
    parser = argparse.ArgumentParser(
        description="Download CMS HRRP data and cache as parquet."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local cache already exists.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Stop after fetching N rows (useful for a quick trial run).",
    )
    args = parser.parse_args()

    try:
        download_hrrp(force=args.force, max_rows=args.max_rows)
    except Exception:
        logger.exception("Ingest failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

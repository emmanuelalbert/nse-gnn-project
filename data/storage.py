"""
data/storage.py
---------------
Low-level Parquet I/O and dataset slice utilities.

All pipeline modules should read/write data through this module rather than
directly calling pd.read_parquet / .to_parquet, so that file paths, engine
choices, and compression settings are managed in one place.

Key helpers:
  - load_prices()       — raw adjusted closing prices
  - load_returns()      — preprocessed log returns (output of preprocessing/)
  - slice_period()      — extract rows for a given shock window
  - save_snapshot()     — versioned backup of any DataFrame
  - list_snapshots()    — list all saved snapshots
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CORE READ / WRITE
# ─────────────────────────────────────────────

def save_df(df: pd.DataFrame, path: Path, label: str = "") -> None:
    """
    Save a DataFrame to Parquet (snappy-compressed, pyarrow engine).
    Creates parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.debug(f"Saved {label or path.name}: {df.shape[0]} rows × {df.shape[1]} cols → {path}")


def load_df(path: Path, label: str = "") -> pd.DataFrame:
    """
    Load a Parquet file and return a DataFrame with a DatetimeIndex.
    Raises FileNotFoundError with a clear message if file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{label or path.name} not found at {path}. "
            "Run the relevant pipeline stage first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    logger.debug(f"Loaded {label or path.name}: {df.shape[0]} rows × {df.shape[1]} cols from {path}")
    return df


# ─────────────────────────────────────────────
# NAMED LOADERS  (thin wrappers over load_df)
# ─────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """
    Load raw adjusted closing prices.
    Shape: (trading_days, n_tickers).
    Produced by: data/collector.py
    """
    return load_df(CFG.RAW_PRICES_FILE, label="raw prices")


def load_returns() -> pd.DataFrame:
    """
    Load preprocessed log returns (KNN-imputed).
    Shape: (trading_days - 1, n_tickers).
    Produced by: preprocessing/returns.py + preprocessing/imputer.py
    """
    return load_df(CFG.LOG_RETURNS_FILE, label="log returns")


def save_prices(df: pd.DataFrame) -> None:
    save_df(df, CFG.RAW_PRICES_FILE, label="raw prices")


def save_returns(df: pd.DataFrame) -> None:
    save_df(df, CFG.LOG_RETURNS_FILE, label="log returns")


# ─────────────────────────────────────────────
# PERIOD SLICING
# ─────────────────────────────────────────────

def slice_period(
    df: pd.DataFrame,
    start: str,
    end: str,
    inclusive: str = "both",
) -> pd.DataFrame:
    """
    Return rows of df whose DatetimeIndex falls within [start, end].

    Args:
        df        : DataFrame with DatetimeIndex.
        start     : ISO date string "YYYY-MM-DD".
        end       : ISO date string "YYYY-MM-DD".
        inclusive : "both" | "left" | "right" | "neither" (pandas convention).

    Returns:
        Sliced DataFrame. Raises ValueError if result is empty.
    """
    mask = pd.Series(True, index=df.index)

    if inclusive in ("both", "left"):
        mask &= df.index >= pd.Timestamp(start)
    else:
        mask &= df.index > pd.Timestamp(start)

    if inclusive in ("both", "right"):
        mask &= df.index <= pd.Timestamp(end)
    else:
        mask &= df.index < pd.Timestamp(end)

    sliced = df.loc[mask]

    if sliced.empty:
        raise ValueError(
            f"slice_period returned 0 rows for [{start}, {end}]. "
            "Check that the date range overlaps with your data."
        )

    return sliced


def slice_shock_period(
    df: pd.DataFrame,
    shock: dict,
    buffer_days: int = 0,
) -> pd.DataFrame:
    """
    Convenience wrapper: slice df for a shock period dict
    (as defined in settings.TARGET_SHOCK_PERIODS).

    Args:
        df          : DataFrame with DatetimeIndex.
        shock       : Dict with keys "start", "end", "name".
        buffer_days : Optional extra days before start and after end.

    Returns:
        Sliced DataFrame for the shock window.
    """
    start = pd.Timestamp(shock["start"]) - pd.Timedelta(days=buffer_days)
    end   = pd.Timestamp(shock["end"])   + pd.Timedelta(days=buffer_days)
    result = slice_period(df, str(start.date()), str(end.date()))
    logger.debug(
        f"Sliced '{shock['name']}': {result.shape[0]} trading days "
        f"({result.index.min().date()} → {result.index.max().date()})"
    )
    return result


def get_date_range(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (first_date, last_date) as ISO strings."""
    return (
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
    )


# ─────────────────────────────────────────────
# VERSIONED SNAPSHOTS
# ─────────────────────────────────────────────

SNAPSHOT_DIR = CFG.DATA_DIR / "_snapshots"


def save_snapshot(df: pd.DataFrame, name: str) -> Path:
    """
    Save a timestamped snapshot of any DataFrame.
    Useful for preserving intermediate states before re-running pipeline stages.

    Snapshot path: data/raw/_snapshots/{name}_{YYYYMMDD_HHMMSS}.parquet
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SNAPSHOT_DIR / f"{name}_{ts}.parquet"
    save_df(df, out_path, label=f"snapshot:{name}")
    logger.info(f"Snapshot saved → {out_path}")
    return out_path


def list_snapshots(name_filter: Optional[str] = None) -> list:
    """
    List all saved snapshots, optionally filtered by name prefix.

    Returns list of dicts: [{name, path, timestamp, size_mb}]
    """
    if not SNAPSHOT_DIR.exists():
        return []

    snapshots = []
    for p in sorted(SNAPSHOT_DIR.glob("*.parquet")):
        if name_filter and not p.stem.startswith(name_filter):
            continue
        snapshots.append({
            "name"      : p.stem,
            "path"      : p,
            "timestamp" : datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size_mb"   : round(p.stat().st_size / 1e6, 2),
        })
    return snapshots


def load_snapshot(name: str) -> pd.DataFrame:
    """
    Load the most recent snapshot matching name prefix.
    Raises FileNotFoundError if no matching snapshot exists.
    """
    matches = list_snapshots(name_filter=name)
    if not matches:
        raise FileNotFoundError(f"No snapshots found with prefix '{name}' in {SNAPSHOT_DIR}")
    latest = matches[-1]   # sorted by filename → chronological
    logger.info(f"Loading snapshot: {latest['name']} ({latest['size_mb']} MB)")
    return load_df(latest["path"])


# ─────────────────────────────────────────────
# DATA INTEGRITY CHECKS
# ─────────────────────────────────────────────

def assert_no_lookahead(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str = "",
) -> None:
    """
    Assert that train and test DataFrames have no temporal overlap.
    Critical for the Monte Carlo chronological splits (paper Section III-D).
    Raises AssertionError if overlap is detected.
    """
    train_end = train_df.index.max()
    test_start = test_df.index.min()

    assert train_end < test_start, (
        f"Data leakage detected {label}: train ends {train_end.date()}, "
        f"test starts {test_start.date()}. These must not overlap."
    )
    logger.debug(f"No lookahead {label}: train ≤ {train_end.date()} < {test_start.date()} ≤ test")


def quick_stats(df: pd.DataFrame, label: str = "") -> None:
    """
    Print a concise data quality report to the logger.
    """
    n_rows, n_cols = df.shape
    missing_pct    = df.isna().mean() * 100
    start, end     = get_date_range(df)

    logger.info(
        f"[{label or 'DataFrame'}] "
        f"{n_rows} rows × {n_cols} cols | "
        f"{start} → {end} | "
        f"missing: avg {missing_pct.mean():.2f}%, max {missing_pct.max():.2f}%"
    )
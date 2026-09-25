"""
config/__init__.py
------------------
Public API for the config package.

Usage anywhere in the project:

    from config import settings, CFG
    from config import get_universe, get_tickers, get_sector_map

    print(CFG.START_DATE)
    print(CFG.TGAT_HIDDEN_DIM)
"""

from config import settings as CFG                        # noqa: F401  — import as module alias

from config.nifty50_universe import (                     # noqa: F401
    get_universe,
    get_tickers,
    get_stock_tickers,
    get_sector_map,
    get_sectors,
    get_stocks_by_sector,
    get_ticker_name_map,
    NIFTY_INDEX,
    SECTOR_COLOURS,
)

# TARGET_SHOCK_PERIODS lives in settings (single source of truth)
from config.settings import TARGET_SHOCK_PERIODS          # noqa: F401

__all__ = [
    "CFG",
    "get_universe",
    "get_tickers",
    "get_stock_tickers",
    "get_sector_map",
    "get_sectors",
    "get_stocks_by_sector",
    "get_ticker_name_map",
    "NIFTY_INDEX",
    "SECTOR_COLOURS",
    "TARGET_SHOCK_PERIODS",
]
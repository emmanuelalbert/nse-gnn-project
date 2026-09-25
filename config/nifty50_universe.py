"""
nifty50_universe.py
-------------------
Canonical list of all 50 entities in our dataset:
  - 49 Nifty 50 constituent stocks (as of 2024 composition, minus HDFC Ltd)
  - 1 Nifty 50 Index itself (^NSEI) — mirrors the paper's treatment of
    the S&P 500 Index as an additional entity alongside the constituents.

NOTE: HDFC Ltd (HDFC.NS) was originally included as a 50th constituent to
match the 2024 Nifty 50 composition, but was removed from this universe.
HDFC Ltd merged into HDFC Bank in July 2023 and Yahoo Finance no longer
serves ANY historical data for the HDFC.NS symbol — not even for the
pre-merger period — so it cannot be fetched via yfinance at all. Rather
than carry a permanently-empty column through the pipeline, it was
dropped from the universe entirely.

Sector taxonomy follows NSE/SEBI classification, adapted to the 10 broad
sectors used in this study for comparative shock analysis.

IMPORTANT — Nifty 50 composition changes over time. This list reflects the
2024 composition (minus HDFC.NS, see above). For historical shock periods
(e.g. 2008 GFC), some tickers may not have been constituents. The pipeline
handles this by using only stocks with sufficient data for each shock
period (see settings.MIN_VALID_DAYS_FRACTION).

Each entry:
  ticker  : Yahoo Finance ticker symbol (NSE suffix ".NS", index "^NSEI")
  name    : Company full name
  sector  : Broad sector for comparative analysis (paper uses ~10 sectors)
  industry: More granular NSE industry classification
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Entity:
    ticker: str
    name: str
    sector: str
    industry: str


# ─────────────────────────────────────────────
# NIFTY 50 INDEX  (the 50th entity, alongside 49 stocks below)
# ─────────────────────────────────────────────

NIFTY_INDEX = Entity(
    ticker   = "^NSEI",
    name     = "Nifty 50 Index",
    sector   = "Index",
    industry = "Benchmark Index",
)


# ─────────────────────────────────────────────
# 50 CONSTITUENT STOCKS
# ─────────────────────────────────────────────

NIFTY50_STOCKS: List[Entity] = [

    # ── INFORMATION TECHNOLOGY ─────────────────
    Entity("INFY.NS",   "Infosys Limited",                 "Information Technology", "IT Services"),
    Entity("TCS.NS",    "Tata Consultancy Services",       "Information Technology", "IT Services"),
    Entity("WIPRO.NS",  "Wipro Limited",                   "Information Technology", "IT Services"),
    Entity("HCLTECH.NS","HCL Technologies",                "Information Technology", "IT Services"),
    Entity("TECHM.NS",  "Tech Mahindra",                   "Information Technology", "IT Services"),
    Entity("LTM.NS",    "LTM Limited (formerly LTIMindtree)", "Information Technology", "IT Services"),

    # ── FINANCIALS ─────────────────────────────
    Entity("HDFCBANK.NS",  "HDFC Bank",                    "Financials", "Private Sector Bank"),
    Entity("ICICIBANK.NS", "ICICI Bank",                   "Financials", "Private Sector Bank"),
    Entity("KOTAKBANK.NS", "Kotak Mahindra Bank",          "Financials", "Private Sector Bank"),
    Entity("AXISBANK.NS",  "Axis Bank",                    "Financials", "Private Sector Bank"),
    Entity("SBIN.NS",      "State Bank of India",          "Financials", "Public Sector Bank"),
    Entity("BAJFINANCE.NS","Bajaj Finance",                 "Financials", "Non-Banking Financial"),
    Entity("BAJAJFINSV.NS","Bajaj Finserv",                 "Financials", "Non-Banking Financial"),
    Entity("SHRIRAMFIN.NS","Shriram Finance",              "Financials", "Non-Banking Financial"),

    # ── ENERGY ─────────────────────────────────
    Entity("RELIANCE.NS",  "Reliance Industries",          "Energy",     "Oil & Gas — Integrated"),
    Entity("ONGC.NS",      "Oil & Natural Gas Corporation","Energy",     "Oil & Gas — Exploration"),
    Entity("BPCL.NS",      "Bharat Petroleum Corp",        "Energy",     "Oil & Gas — Refining"),
    Entity("POWERGRID.NS", "Power Grid Corporation",       "Energy",     "Power Transmission"),
    Entity("NTPC.NS",      "NTPC Limited",                 "Energy",     "Power Generation"),
    Entity("COALINDIA.NS", "Coal India",                   "Energy",     "Mining — Coal"),
    Entity("ADANIENT.NS",  "Adani Enterprises",            "Energy",     "Conglomerate — Energy"),

    # ── FMCG ───────────────────────────────────
    Entity("HINDUNILVR.NS","Hindustan Unilever",           "FMCG",       "Personal & Home Care"),
    Entity("ITC.NS",       "ITC Limited",                  "FMCG",       "Cigarettes & FMCG"),
    Entity("NESTLEIND.NS", "Nestle India",                 "FMCG",       "Food & Beverages"),
    Entity("BRITANNIA.NS", "Britannia Industries",         "FMCG",       "Food & Beverages"),

    # ── PHARMACEUTICALS ────────────────────────
    Entity("SUNPHARMA.NS", "Sun Pharmaceutical",           "Pharma",     "Pharmaceuticals"),
    Entity("DRREDDY.NS",   "Dr. Reddy's Laboratories",    "Pharma",     "Pharmaceuticals"),
    Entity("CIPLA.NS",     "Cipla Limited",                "Pharma",     "Pharmaceuticals"),
    Entity("DIVISLAB.NS",  "Divi's Laboratories",          "Pharma",     "Pharmaceuticals"),
    Entity("APOLLOHOSP.NS","Apollo Hospitals",             "Pharma",     "Healthcare Services"),

    # ── METALS & MINING ────────────────────────
    Entity("TATASTEEL.NS", "Tata Steel",                   "Metals",     "Steel"),
    Entity("JSWSTEEL.NS",  "JSW Steel",                    "Metals",     "Steel"),
    Entity("HINDALCO.NS",  "Hindalco Industries",          "Metals",     "Aluminium"),
    Entity("VEDL.NS",      "Vedanta Limited",              "Metals",     "Diversified Metals"),

    # ── AUTOMOBILE ─────────────────────────────
    Entity("MARUTI.NS",    "Maruti Suzuki India",          "Automobile", "Passenger Vehicles"),
    Entity("M&M.NS",       "Mahindra & Mahindra",          "Automobile", "Passenger Vehicles"),
    Entity("TMPV.NS",      "Tata Motors",                  "Automobile", "Commercial Vehicles"),  # renamed from TATAMOTORS.NS after Oct 2025 demerger; same continuous listing, full 2005-present history
    Entity("BAJAJ-AUTO.NS","Bajaj Auto",                   "Automobile", "Two-Wheelers"),
    Entity("EICHERMOT.NS", "Eicher Motors",                "Automobile", "Two-Wheelers"),
    Entity("HEROMOTOCO.NS","Hero MotoCorp",                "Automobile", "Two-Wheelers"),

    # ── CONSUMER DISCRETIONARY ─────────────────
    Entity("TITAN.NS",     "Titan Company",                "Consumer Discretionary", "Jewellery & Watches"),
    Entity("ASIANPAINT.NS","Asian Paints",                 "Consumer Discretionary", "Paints"),
    Entity("INDUSINDBK.NS","IndusInd Bank",                "Consumer Discretionary", "Private Sector Bank"),  # reclassified for analysis

    # ── TELECOM ────────────────────────────────
    Entity("BHARTIARTL.NS","Bharti Airtel",               "Telecom",    "Telecom Services"),

    # ── CEMENT & INFRASTRUCTURE ────────────────
    Entity("ULTRACEMCO.NS","UltraTech Cement",            "Cement & Infra", "Cement"),
    Entity("GRASIM.NS",    "Grasim Industries",            "Cement & Infra", "Cement & Fibres"),
    Entity("ADANIPORTS.NS","Adani Ports & SEZ",           "Cement & Infra", "Ports & Logistics"),

    # ── CONGLOMERATE ───────────────────────────
    Entity("LT.NS",        "Larsen & Toubro",              "Conglomerate", "Engineering & Construction"),
    Entity("SIEMENS.NS",   "Siemens India",                "Conglomerate", "Engineering & Industrials"),
]


# ─────────────────────────────────────────────
# COMBINED UNIVERSE  (50 entities)
# ─────────────────────────────────────────────

def get_universe() -> List[Entity]:
    """Return deduplicated list of all 50 entities (index + 49 stocks)."""
    seen = set()
    unique = []
    for entity in [NIFTY_INDEX] + NIFTY50_STOCKS:
        if entity.ticker not in seen:
            seen.add(entity.ticker)
            unique.append(entity)
    return unique


def get_tickers() -> List[str]:
    """Return list of all ticker symbols."""
    return [e.ticker for e in get_universe()]


def get_stock_tickers() -> List[str]:
    """Return only stock tickers (exclude index)."""
    return [e.ticker for e in get_universe() if e.ticker != "^NSEI"]


def get_sector_map() -> dict:
    """Return {ticker: sector} mapping for all entities."""
    return {e.ticker: e.sector for e in get_universe()}


def get_sectors() -> List[str]:
    """Return sorted list of unique sector names."""
    return sorted(set(e.sector for e in get_universe()))


def get_stocks_by_sector(sector: str) -> List[Entity]:
    """Return all entities belonging to a given sector."""
    return [e for e in get_universe() if e.sector == sector]


def get_ticker_name_map() -> dict:
    """Return {ticker: company_name} for labels and plots."""
    return {e.ticker: e.name for e in get_universe()}


# ─────────────────────────────────────────────
# SECTOR COLOURS  (used in all visualisations)
# ─────────────────────────────────────────────

SECTOR_COLOURS = {
    "Index":                    "#374151",
    "Information Technology":   "#6366f1",
    "Financials":               "#0ea5e9",
    "Energy":                   "#f59e0b",
    "FMCG":                     "#10b981",
    "Pharma":                   "#ec4899",
    "Metals":                   "#8b5cf6",
    "Automobile":               "#f97316",
    "Consumer Discretionary":   "#14b8a6",
    "Telecom":                  "#06b6d4",
    "Cement & Infra":           "#a3a3a3",
    "Conglomerate":             "#64748b",
}


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    universe = get_universe()
    print(f"Total entities  : {len(universe)}")
    print(f"Stocks          : {len(get_stock_tickers())}")
    print(f"Sectors         : {get_sectors()}")
    print()
    for sector in get_sectors():
        stocks = get_stocks_by_sector(sector)
        print(f"  {sector:<28} {len(stocks)} entities")
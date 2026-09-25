"""
settings.py
-----------
Central configuration for the Nifty 50 TGAT Shock Propagation project.

All hyperparameters, paths, and constants live here. Import from this module
everywhere else — never hardcode values in individual modules.

Reference: Bentaleb et al. (2024), IEEE Access, DOI 10.1109/ACCESS.2024.3487766
Adaptation: S&P 500 (498 entities) → Nifty 50 (50 entities, NSE India)
"""

from pathlib import Path
import torch

# ─────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parent.parent
CONFIG_DIR  = ROOT_DIR / "config"
DATA_DIR    = ROOT_DIR / "data" / "raw"
RESULTS_DIR = ROOT_DIR / "results"

METRICS_DIR     = RESULTS_DIR / "metrics"
ADJACENCY_DIR   = RESULTS_DIR / "adjacency"
CHECKPOINT_DIR  = RESULTS_DIR / "checkpoints"

# Which metric decides the "best" checkpoint saved per fold, and whether
# higher or lower is better for it.
#
# Default is "auc_roc"/"max" rather than "val_loss"/"min" because we found
# empirically that they diverge on this task: a checkpoint with the lowest
# val_loss (epoch 1, essentially pre-training) scored AUC-ROC=0.39, while a
# later epoch with a WORSE val_loss (0.85-0.93 vs 0.58) scored AUC-ROC=
# 0.71-0.74 and a much higher F1 (0.30 vs 0.00). Loss-based selection was
# silently discarding the epoch that had actually learned something useful.
# Since the paper's own reported results (Eq. 6-8) are AUC-ROC/AUPR/F1, not
# raw loss, selecting by AUC-ROC is the more defensible default. Set back
# to CHECKPOINT_MONITOR="val_loss", CHECKPOINT_MODE="min" to restore the
# original behaviour if needed for comparison.
CHECKPOINT_MONITOR = "auc_roc"
CHECKPOINT_MODE     = "max"
FIGURES_DIR     = RESULTS_DIR / "figures"

# ─────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────

# 2005-01-01: pre-2005 NSE data via yfinance has quality gaps.
# This range captures: 2008 GFC, 2011 sovereign debt, 2013 taper tantrum,
# 2016 demonetization, 2018 IL&FS, 2020 COVID, 2022 FII selloff, 2023 Adani.
START_DATE = "2005-01-01"
END_DATE   = "2025-06-01"   # pull to present; update periodically

# Yahoo Finance tickers for all 50 Nifty constituents + index.
# Nifty 50 index itself is treated as an additional entity (mirrors paper's 498 = 497 + index).
NIFTY_INDEX_TICKER = "^NSEI"

# Parquet filenames
RAW_PRICES_FILE  = DATA_DIR / "nifty50_closing_prices.parquet"
LOG_RETURNS_FILE = DATA_DIR / "nifty50_log_returns.parquet"

# yfinance download settings
YFINANCE_INTERVAL = "1d"          # daily closing prices
YFINANCE_AUTO_ADJUST = True       # adjust for splits and dividends
YFINANCE_PROGRESS = False         # suppress per-ticker progress bars

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

# KNN imputer for missing values (paper Section III-A-2c).
# KNN preserves inter-stock relationships better than mean/zero imputation.
KNN_N_NEIGHBORS = 5

# Minimum number of non-NaN trading days required to keep a stock in the dataset.
# Stocks with more missing data than this threshold are dropped entirely.
MIN_VALID_DAYS_FRACTION = 0.90   # 90% of total trading days must be present

# ─────────────────────────────────────────────
# SHOCK DETECTION  (paper Eq. 2)
# ─────────────────────────────────────────────

# A shock period is identified if the absolute value of the average daily
# log return over a rolling window exceeds SHOCK_SIGMA_THRESHOLD standard
# deviations from the long-term mean.
SHOCK_WINDOW_DAYS       = 5      # minimum consecutive trading days (paper: 5)
SHOCK_SIGMA_THRESHOLD   = 2.0   # |avg_return - mu| > 2 * sigma

# SIGMA BASIS — this is a genuine ambiguity in the paper's wording (Eq. 2),
# not a settled choice, so it's exposed as a toggle rather than hardcoded:
#
#   "daily"   : sigma = std of individual daily log returns (the literal
#               reading of the paper's text). By the CLT, a 5-day *average*
#               naturally has a much tighter spread than single days
#               (~sigma_daily / sqrt(5)), so comparing a 5-day average
#               against 2 * sigma_daily is a very strict bar in practice —
#               empirically this produced only ~7 shocks over 20 years on
#               Nifty 50, versus the paper's reported 53 over 23 years on
#               the S&P 500. Likely too conservative to be what the paper's
#               authors actually ran, even though it matches the text.
#
#   "rolling" : sigma = std of the rolling SHOCK_WINDOW_DAYS-average return
#               series itself. This is the more standard convention for
#               this kind of test and is much more likely to reproduce a
#               paper-comparable shock count (30-55 over a similar span).
#
# Default is "rolling" — if you want the literal-text reading instead,
# switch to "daily" and expect a substantially lower shock count.
SHOCK_SIGMA_BASIS       = "rolling"   # "rolling" or "daily"

# The Nifty 50 Index (^NSEI) is used as the reference series for computing
# the long-term mean (mu) and standard deviation (sigma).
SHOCK_REFERENCE_TICKER  = NIFTY_INDEX_TICKER

# Merge shock windows that are fewer than this many days apart.
# Prevents detecting the same crisis event as multiple separate shocks.
SHOCK_MERGE_GAP_DAYS    = 3

# ─────────────────────────────────────────────
# GRANGER CAUSALITY  (paper Eq. 3 & 4)
# ─────────────────────────────────────────────

# Significance threshold for rejecting the null hypothesis
# (X does not Granger-cause Y). Paper uses p <= 0.05.
GRANGER_P_THRESHOLD = 0.05

# Test statistic: 'ssr_chi2test' (paper choice) — robust sum-of-squared-
# residuals chi-squared test. Alternatives: 'ssr_ftest', 'lrtest', 'params_ftest'.
GRANGER_TEST        = "ssr_chi2test"

# Adaptive lag: for each shock period, max lag is bounded by period length.
# Set GRANGER_MAX_LAG = None to use adaptive (recommended).
# Paper: max lag ranges from 1 day up to the length of the period.
GRANGER_MAX_LAG     = None       # None → adaptive per period
GRANGER_MIN_LAG     = 1

# Number of parallel jobs for Granger computation.
# With 49 stocks → 49×49 = 2,401 pairs per period. Parallelise across pairs.
# Set -1 to use all available CPU cores.
GRANGER_N_JOBS      = -1

# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────

# Directed graph: edge A→B exists if A Granger-causes B (GC coefficient = 1).
# Self-loops for intra-company are set to 1 (paper Section III-A-4).
GRAPH_DIRECTED         = True
GRAPH_SELF_LOOPS       = True

# Link score threshold for the TGAT-predicted adjacency matrix.
# Edges with predicted probability >= this value are considered active.
GRAPH_LINK_THRESHOLD   = 0.5

# THRESHOLD METHOD for converting predicted probabilities into a binary
# adjacency matrix in stage_postadj. Two options:
#
#   "absolute"   : edge exists if prob >= GRAPH_LINK_THRESHOLD. Only
#                  meaningful if the decoder's output is well-calibrated
#                  across the full probability range. With limited training
#                  data (few shock periods / few epochs before early
#                  stopping), the decoder can end up producing predictions
#                  crammed into a razor-thin band just above or below 0.5
#                  (e.g. empirically observed: min=0.512, max=0.529,
#                  std=0.0037 on this project's 3-epoch/1-fold run) — every
#                  single pair trivially clears an absolute 0.5 cutoff,
#                  producing a degenerate 100% density matrix even though
#                  the model has real, useful RELATIVE ranking ability
#                  (that same run scored AUC-ROC=0.75).
#
#   "percentile" : edge exists if prob is in the top GRAPH_LINK_TOP_PCT
#                  percent for that period. This uses the model's relative
#                  ranking (which is what AUC-ROC/AUPR actually measure)
#                  instead of its uncalibrated absolute output, and is the
#                  more defensible choice when the decoder hasn't had
#                  enough training to spread predictions across the full
#                  (0,1) range.
#
# Default is "percentile" for the reasons above. Revisit once training has
# substantially more epochs/data and absolute calibration can be trusted.
GRAPH_LINK_METHOD      = "percentile"
GRAPH_LINK_TOP_PCT     = 30.0   # top 30% most-confident pairs become edges

# Node feature used in the paper: mean of log returns over the shock window.
# Additional features can be added per NODE_FEATURE_COLS below.
NODE_FEATURE_COLS = ["mean_return"]   # extend to e.g. ["mean_return", "volatility", "volume_zscore"]

# ─────────────────────────────────────────────
# TGAT MODEL ARCHITECTURE  (paper Section III-B)
# ─────────────────────────────────────────────

# Temporal Graph Attention Network parameters.
# TGAT improvement over paper's TGCN: multi-head attention with learnable
# time encoding (cos/sin positional encoding on time deltas) rather than
# a simple scalar temporal attention coefficient.

TGAT_NUM_LAYERS         = 2       # number of TGAT message-passing layers
TGAT_HIDDEN_DIM         = 64      # hidden node embedding dimension
TGAT_NUM_HEADS          = 4       # multi-head attention heads
TGAT_TIME_DIM           = 16      # dimension of time encoding vector
TGAT_DROPOUT            = 0.3     # dropout rate (paper: 0.3)
TGAT_USE_LAYER_NORM     = True    # LayerNorm after each TGAT layer (paper)
TGAT_RESIDUAL           = True    # residual connections after each layer pair (paper)

# Decoder: linear layer on concatenated node embeddings → link score
TGAT_DECODER_HIDDEN_DIM = 32      # intermediate dim in link decoder MLP

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

TRAIN_EPOCHS            = 200
TRAIN_BATCH_SIZE        = 1       # one shock period graph per batch (small dataset)
TRAIN_LR                = 1e-3
TRAIN_L2_WEIGHT_DECAY   = 1e-5   # L2 regularisation (paper: 1e-5)
TRAIN_EARLY_STOP_PATIENCE = 20   # stop if val loss doesn't improve for N epochs
TRAIN_EARLY_STOP_DELTA  = 1e-4   # minimum improvement to count as progress

# Negative sampling ratio: number of negative edges per positive edge.
# Paper uses balanced sampling (ratio ≈ 1.0).
NEGATIVE_SAMPLING_RATIO = 1.0

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)   # auto-detected — was previously hardcoded to "cpu", silently ignoring an
    # available GPU (RTX 3050 confirmed present via nvidia-smi + CUDA-enabled
    # torch install). Override manually here if you ever need to force CPU
    # for debugging (e.g. DEVICE = "cpu").

# ─────────────────────────────────────────────
# MONTE CARLO EVALUATION  (paper Section III-D, Eq. 11 & 12)
# ─────────────────────────────────────────────

# Chronological splits: shock periods are sorted by date, then divided
# into 80% training / 20% test without any shuffling (no data leakage).
MONTE_CARLO_N_SPLITS    = 10
TRAIN_RATIO             = 0.80

# Random seed for reproducibility (negative sampling, weight init).
RANDOM_SEED             = 42

# ─────────────────────────────────────────────
# NETWORK ANALYSIS METRICS  (paper Section III-E, Eq. 13–20)
# ─────────────────────────────────────────────

# All centrality metrics are computed on both the Granger adjacency matrix
# (pre-training) and the TGAT-predicted adjacency matrix (post-training).
# The delta between them is the core analytical finding.

CENTRALITY_METRICS = [
    "degree",           # Eq. 13  — number of connections per node
    "closeness",        # Eq. 14  — inverse average shortest path
    "betweenness",      # Eq. 15  — fraction of shortest paths through node
    "degree_centrality",# Eq. 18  — normalised degree
]

# Amplitude and average change computed per stock and per sector (Eq. 19–20).
COMPUTE_SECTOR_METRICS = True

# ─────────────────────────────────────────────
# COMPARATIVE ANALYSIS: TARGET SHOCK PERIODS
# ─────────────────────────────────────────────

# Three India-specific shock periods chosen to mirror the paper's structure:
#   Paper:    2008 GFC  |  2016 US Elections  |  COVID-19 onset
#   Ours:     2008 GFC  |  2018 IL&FS crisis  |  COVID-19 onset
#
# A fourth optional period (demonetization) is unique to the Indian market.
# These are starting points; the shock detector will refine exact dates.

TARGET_SHOCK_PERIODS = [
    {
        "name": "Global Financial Crisis",
        "start": "2008-09-15",   # Lehman Brothers collapse
        "end":   "2008-10-31",
        "type":  "global_financial",
    },
    {
        "name": "IL&FS and NBFC Liquidity Crisis",
        "start": "2018-09-21",   # IL&FS default
        "end":   "2018-10-26",
        "type":  "domestic_financial",
    },
    {
        "name": "COVID-19 Pandemic Onset",
        "start": "2020-02-20",   # mirrors paper exactly
        "end":   "2020-03-23",
        "type":  "health_nonfinancial",
    },
    {
        "name": "Demonetization Shock",   # optional — unique India policy shock
        "start": "2016-11-08",
        "end":   "2016-11-25",
        "type":  "policy",
    },
]

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE   = ROOT_DIR / "pipeline.log"
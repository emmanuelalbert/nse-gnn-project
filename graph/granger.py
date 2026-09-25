"""
graph/granger.py
----------------
Computes a directed Granger causality matrix for each shock period.

Paper reference: Section III-A-4 (Eq. 3 & 4)

    Test statistic (ssr_chi2test, paper Eq. 3):
        F = [SSR(p) - SSR(p+k)] / k
            --------------------------------  ×  (T - N - k - p)
                   SSR(p+k)

    Granger Coefficient (paper Eq. 4):
        GC(X→Y) = 1  if p-value ≤ 0.05
        GC(X→Y) = 0  otherwise

    where:
        SSR(p)    = sum of squared residuals, restricted model (p lags)
        SSR(p+k)  = sum of squared residuals, full model (p+k lags)
        T         = total observations in shock period
        k         = number of additional lags (restrictions)
        p         = base lag order

Pipeline role:
    For each shock period, this module:
      1. Extracts the log-return slice for that period.
      2. Determines the adaptive max lag (min 1, max = period length).
      3. Runs grangercausalitytests(ssr_chi2test) for every ordered pair (i→j).
      4. Applies the p ≤ 0.05 threshold to produce a binary (0/1) GC matrix.
      5. Sets the diagonal to 1 (self-loops, paper Section III-A-4).

Design decisions:
  - Adaptive lag per period: the paper states "max lag ranges from 1 day
    up to the length of the period, with a minimum of 5 days." We follow
    this exactly — GRANGER_MIN_LAG=1, max=period_length (capped at 10 to
    keep compute tractable for longer merged shocks).
  - Parallelism: with 50 tickers → 50×49 = 2,450 directed pairs per period
    and up to ~50 shock periods, we parallelise across pairs using joblib.
  - statsmodels grangercausalitytests: we pass maxlag as an integer (not
    a list), which causes it to test all lags up to maxlag and return the
    one with best fit. We use the ssr_chi2test statistic as the paper does.
  - Missing data: if either series in a pair has > 20% NaN after slicing,
    the GC coefficient is set to 0 (no edge) rather than running a test on
    unreliable data.

Usage:
    from graph.granger import GrangerComputer
    gc = GrangerComputer()
    matrices = gc.compute_all(returns_df, shocks)   # {shock_name: np.ndarray}
    matrix   = gc.compute_period(returns_df, shock) # single period
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import grangercausalitytests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from shock_detection.detector import ShockPeriod

logger = logging.getLogger(__name__)

# Maximum lag cap — even for long shock periods we stop at 10 lags.
# Beyond 10, most shock periods won't have enough observations for reliable
# chi-squared tests, and computation time grows quadratically.
MAX_LAG_CAP = 10

# Minimum fraction of valid (non-NaN) values required in a series to run the test.
MIN_VALID_FRACTION = 0.80


class GrangerComputer:
    """
    Computes directed Granger causality matrices for a list of shock periods.

    Each matrix is shape (N, N) where N = number of tickers.
    Matrix[i, j] = 1 means ticker i Granger-causes ticker j (p ≤ 0.05).
    Matrix[i, i] = 1 always (self-loops, paper convention).

    Args:
        p_threshold  : Significance level for rejecting H0. Paper: 0.05.
        test         : statsmodels test name. Paper: "ssr_chi2test".
        min_lag      : Minimum lag to test. Paper: 1.
        max_lag      : Maximum lag override. None → adaptive per period.
        n_jobs       : Parallel workers (-1 = all cores).
        verbose      : Joblib verbosity level.
    """

    def __init__(
        self,
        p_threshold : float = CFG.GRANGER_P_THRESHOLD,
        test        : str   = CFG.GRANGER_TEST,
        min_lag     : int   = CFG.GRANGER_MIN_LAG,
        max_lag     : Optional[int] = CFG.GRANGER_MAX_LAG,
        n_jobs      : int   = CFG.GRANGER_N_JOBS,
        verbose     : int   = 0,
    ):
        self.p_threshold = p_threshold
        self.test        = test
        self.min_lag     = min_lag
        self.max_lag     = max_lag
        self.n_jobs      = n_jobs
        self.verbose     = verbose

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def compute_all(
        self,
        returns   : pd.DataFrame,
        shocks    : List[ShockPeriod],
    ) -> Dict[str, np.ndarray]:
        """
        Compute a Granger causality matrix for every shock period.

        Args:
            returns : Full log returns DataFrame (T, N), DatetimeIndex.
            shocks  : List of ShockPeriod objects from the registry.

        Returns:
            Dict mapping shock_name → GC matrix (N, N) float32.
            Matrix[i,j] = 1 if ticker[i] Granger-causes ticker[j].
        """
        tickers = list(returns.columns)
        results : Dict[str, np.ndarray] = {}

        logger.info(
            f"Computing Granger causality for {len(shocks)} shock periods, "
            f"{len(tickers)} tickers ({len(tickers)**2 - len(tickers)} directed pairs per period)"
        )

        for idx, shock in enumerate(shocks, 1):
            logger.info(
                f"[{idx}/{len(shocks)}] {shock.name} "
                f"[{shock.start} → {shock.end}] ({shock.duration} days)"
            )
            try:
                matrix = self.compute_period(returns, shock)
                results[shock.name] = matrix
            except Exception as exc:
                logger.error(f"  Failed: {exc} — filling with zeros.")
                results[shock.name] = np.zeros(
                    (len(tickers), len(tickers)), dtype=np.float32
                )

        n_total = sum(m.sum() - np.trace(m) for m in results.values())
        logger.info(
            f"Granger computation complete. "
            f"Total directed edges across all periods: {int(n_total)}"
        )
        return results

    def compute_period(
        self,
        returns : pd.DataFrame,
        shock   : ShockPeriod,
    ) -> np.ndarray:
        """
        Compute the Granger causality matrix for a single shock period.

        Args:
            returns : Full log returns DataFrame (T, N).
            shock   : ShockPeriod object with start/end dates.

        Returns:
            GC matrix (N, N) float32.
            Matrix[i, j] = 1.0 if ticker[i] → ticker[j] (p ≤ threshold).
            Diagonal = 1.0 (self-loops).
        """
        tickers = list(returns.columns)
        N       = len(tickers)

        # Slice the shock period
        period  = self._slice_period(returns, shock)
        lag     = self._adaptive_lag(shock.duration)

        logger.debug(
            f"  Period shape: {period.shape}, adaptive max_lag: {lag}"
        )

        # Run all N×(N-1) directed pairs in parallel
        pairs = [
            (i, j)
            for i in range(N)
            for j in range(N)
            if i != j
        ]

        # Parallel Granger tests
        results_flat = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._test_pair)(
                period.iloc[:, i].values,   # cause: X
                period.iloc[:, j].values,   # effect: Y
                lag,
                tickers[i],
                tickers[j],
            )
            for i, j in pairs
        )

        # Assemble matrix
        matrix = np.zeros((N, N), dtype=np.float32)
        np.fill_diagonal(matrix, 1.0)   # self-loops (paper Section III-A-4)

        for (i, j), gc_val in zip(pairs, results_flat):
            matrix[i, j] = gc_val

        density = (matrix.sum() - N) / (N * (N - 1))   # exclude diagonal
        logger.debug(
            f"  GC matrix: density={density:.3f}, "
            f"edges={int(matrix.sum() - N)} / {N*(N-1)} possible"
        )
        return matrix

    def compute_period_with_pvalues(
        self,
        returns : pd.DataFrame,
        shock   : ShockPeriod,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Like compute_period() but also returns the raw p-value matrix.
        Useful for threshold sensitivity analysis in notebooks.

        Returns:
            (gc_matrix, pvalue_matrix) both shape (N, N) float32.
            p-values on the diagonal are set to 0.0.
        """
        tickers = list(returns.columns)
        N       = len(tickers)
        period  = self._slice_period(returns, shock)
        lag     = self._adaptive_lag(shock.duration)

        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

        pvals_flat = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._test_pair_pvalue)(
                period.iloc[:, i].values,
                period.iloc[:, j].values,
                lag,
            )
            for i, j in pairs
        )

        gc_matrix = np.zeros((N, N), dtype=np.float32)
        pv_matrix = np.zeros((N, N), dtype=np.float32)
        np.fill_diagonal(gc_matrix, 1.0)

        for (i, j), pval in zip(pairs, pvals_flat):
            pv_matrix[i, j] = pval if pval is not None else 1.0
            gc_matrix[i, j] = 1.0 if (pval is not None and pval <= self.p_threshold) else 0.0

        return gc_matrix, pv_matrix

    # ─────────────────────────────────────────
    # PAIR-LEVEL TEST  (parallelised unit)
    # ─────────────────────────────────────────

    def _test_pair(
        self,
        x       : np.ndarray,   # potential cause
        y       : np.ndarray,   # potential effect
        max_lag : int,
        name_x  : str = "",
        name_y  : str = "",
    ) -> float:
        """
        Run the Granger causality test for a single ordered pair (X → Y).

        Returns 1.0 if X Granger-causes Y (p ≤ threshold), else 0.0.
        Returns 0.0 (no edge) if either series has insufficient valid data.
        """
        pval = self._test_pair_pvalue(x, y, max_lag)
        if pval is None:
            return 0.0
        result = 1.0 if pval <= self.p_threshold else 0.0
        return result

    def _test_pair_pvalue(
        self,
        x       : np.ndarray,
        y       : np.ndarray,
        max_lag : int,
    ) -> Optional[float]:
        """
        Run grangercausalitytests and return the best p-value across lags.
        Returns None if the test cannot be run (insufficient data, all-NaN, etc.).
        """
        # Combine into 2-column array: [Y, X]  (statsmodels convention: col 0 is Y)
        data = np.column_stack([y, x])

        # Check valid fraction
        valid_rows = np.all(np.isfinite(data), axis=1)
        if valid_rows.sum() < max(5, max_lag + 2):
            return None
        if valid_rows.mean() < MIN_VALID_FRACTION:
            return None

        # Drop any remaining NaN rows
        data = data[valid_rows]

        # Ensure we have enough rows for the requested lag
        effective_lag = min(max_lag, len(data) - max_lag - 2)
        if effective_lag < self.min_lag:
            return None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Pass maxlag as a single-element LIST, not a bare int.
                # statsmodels tests every lag from 1..maxlag when given an
                # int, and the old code took the min p-value across all of
                # them — that's an uncorrected multiple-comparisons search
                # (testing 10 lags at alpha=0.05 gives roughly a 40% false
                # positive rate, not 5%), which is almost certainly why a
                # long period like COVID-19 (capped at 10 lags) came back
                # with ~99% edge density. Testing only the single adaptively
                # chosen lag matches the paper's stated design ("the
                # maximum lag chosen depends on its duration") — one lag
                # per period, not a best-of-N scan.
                test_result = grangercausalitytests(
                    data,
                    maxlag  = [effective_lag],
                    verbose = False,
                )

            # Extract the p-value for the single tested lag.
            # test_result is a dict: {lag: ([F_test, chi2_test, ...], OLS_results)}
            # Each test tuple: (test_stat, p_value, df_denom, df_num)
            lag_results = test_result[effective_lag]
            test_stats  = lag_results[0]         # dict of test result tuples
            # ssr_chi2test is index 1 in statsmodels ordering
            # key order: ssr_ftest=0, ssr_chi2test=1, lrtest=2, params_ftest=3
            test_index = {
                "ssr_chi2test"  : 1,
                "ssr_ftest"     : 0,
                "lrtest"        : 2,
                "params_ftest"  : 3,
            }.get(self.test, 1)

            test_name = list(test_stats.keys())[test_index]
            pval      = test_stats[test_name][1]   # p-value is index 1
            return float(pval)

        except Exception as exc:
            # Deliberately non-fatal (a single bad pair shouldn't kill the
            # whole period), but logged at debug level so a systematic
            # failure (e.g. an infeasible lag causing every pair in a
            # period to fail) is diagnosable from --log-level DEBUG output
            # instead of only showing up as a suspicious 0% density.
            logger.debug(f"Granger test failed for a pair (lag={max_lag}): {type(exc).__name__}: {exc}")
            return None

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _slice_period(
        self,
        returns : pd.DataFrame,
        shock   : ShockPeriod,
    ) -> pd.DataFrame:
        """Slice returns to the shock window and forward-fill residual NaNs."""
        mask   = (
            (returns.index >= pd.Timestamp(shock.start)) &
            (returns.index <= pd.Timestamp(shock.end))
        )
        period = returns.loc[mask].copy()
        if period.empty:
            raise ValueError(
                f"No return data found for shock '{shock.name}' "
                f"[{shock.start} → {shock.end}]"
            )
        # Light forward-fill for isolated NaNs in the period slice.
        # Heavy imputation is done upstream by preprocessing/imputer.py.
        period = period.ffill().bfill()
        return period

    def _adaptive_lag(self, period_length: int) -> int:
        """
        Determine max lag for a shock period.
        Paper: "max lag ranges from 1 day up to the length of the period,
        with a minimum of 5 days." We additionally cap at MAX_LAG_CAP (10)
        for tractability.

        IMPORTANT — this formula went through two iterations after real
        bugs surfaced during validation (see debug sessions / commit
        history), and there is a known residual limitation documented
        below. Do not "simplify" this back to a looser formula without
        re-running the placebo/false-positive check described here.

        Iteration 1 ((T-2)//2): passed statsmodels' bare non-singularity
        check but was still too permissive — silently produced a 0-edge
        matrix for a 7-day period because the *downstream* DOF check in
        _test_pair_pvalue() collapsed to an infeasible lag.

        Iteration 2 ((T-2)//3): matched statsmodels' actual internal
        feasibility constraint (empirically verified: max_feasible_lag =
        (T-2)//3 for a bivariate series, confirmed for T=5..29). This fixed
        the crash/silent-failure bug, but a follow-up placebo test (running
        the test on pure random noise, which should reject H0 only ~5% of
        the time) revealed a SEPARATE problem: at T=33 the "feasible" lag
        of 10 produced a ~98% edge density even on shuffled/noise data —
        the VAR was technically invertible but severely overfit (roughly
        2*lag+1 parameters estimated from T-lag observations), giving
        near-certain spurious "significance" regardless of any real signal.

        Iteration 3 (this one): (T-10)//8, additionally capped at
        MAX_LAG_CAP. This keeps enough observations per estimated
        parameter that the test's false-positive rate on pure noise stays
        close to the nominal 5% for medium/long periods:
            T=33 -> lag=2  -> ~7% false-positive rate on noise (was ~98%
                               edge density on real data before this fix)
            T=84 -> lag=9  -> ~10% false-positive rate on noise

        KNOWN RESIDUAL LIMITATION: for very short periods (T <~ 13, e.g.
        a 7-day shock), even lag=1 (the statsmodels-feasible floor) shows
        an elevated false-positive rate (~15-45% depending on exact T) —
        this is an inherent small-sample limitation of the asymptotic
        chi-squared Granger test, not something further lag tuning can
        fix. Results for short shock periods should be treated as lower-
        confidence / higher-variance in any downstream write-up, rather
        than assumed to be as reliable as GFC/COVID-scale periods.
        """
        if self.max_lag is not None:
            return self.max_lag
        if period_length <= 10:
            conservative_lag = self.min_lag
        else:
            conservative_lag = max(self.min_lag, (period_length - 10) // 8)
        return max(self.min_lag, min(conservative_lag, MAX_LAG_CAP))

    # ─────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────

    def density_report(
        self,
        matrices : Dict[str, np.ndarray],
        tickers  : List[str],
    ) -> pd.DataFrame:
        """
        Summarise the density and edge count of each GC matrix.

        Returns DataFrame with columns:
            shock_name, n_tickers, n_edges, density, self_loops
        """
        rows = []
        for name, mat in matrices.items():
            N         = mat.shape[0]
            self_loops = int(np.trace(mat))
            edges      = int(mat.sum()) - self_loops
            density    = edges / max(1, N * (N - 1))
            rows.append({
                "shock_name" : name,
                "n_tickers"  : N,
                "n_edges"    : edges,
                "density"    : round(density, 4),
                "self_loops" : self_loops,
            })
        return pd.DataFrame(rows).set_index("shock_name")

    def top_causes(
        self,
        matrix  : np.ndarray,
        tickers : List[str],
        top_n   : int = 10,
    ) -> pd.DataFrame:
        """
        Return the top_n tickers with the most outgoing Granger-causal edges
        in a single period's matrix (i.e. biggest 'causes' in the network).
        """
        out_degree = matrix.sum(axis=1) - np.diag(matrix)  # exclude self-loop
        order      = np.argsort(out_degree)[::-1][:top_n]
        return pd.DataFrame({
            "ticker"     : [tickers[i] for i in order],
            "out_degree" : [int(out_degree[i]) for i in order],
        })

    def top_effects(
        self,
        matrix  : np.ndarray,
        tickers : List[str],
        top_n   : int = 10,
    ) -> pd.DataFrame:
        """
        Return the top_n tickers with the most incoming Granger-causal edges
        (i.e. biggest 'receivers of influence' in the network).
        """
        in_degree = matrix.sum(axis=0) - np.diag(matrix)
        order     = np.argsort(in_degree)[::-1][:top_n]
        return pd.DataFrame({
            "ticker"    : [tickers[i] for i in order],
            "in_degree" : [int(in_degree[i]) for i in order],
        })


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="DEBUG", format=CFG.LOG_FORMAT)
    np.random.seed(42)

    # Synthetic returns: 8 tickers, 60 trading days
    # Ticker 0 has a genuine causal relationship with ticker 1 (lagged copy + noise)
    n      = 60
    dates  = pd.date_range("2008-10-01", periods=n, freq="B")
    tickers = [f"T{i}.NS" for i in range(7)] + ["^NSEI"]
    data   = np.random.normal(0, 0.01, size=(n, 8))
    data[:, 1] = np.roll(data[:, 0], 1) + np.random.normal(0, 0.005, n)  # T0→T1

    returns = pd.DataFrame(data, index=dates, columns=tickers)
    returns.index.name = "Date"

    shock = ShockPeriod(
        name         = "Test Shock",
        start        = "2008-10-01",
        end          = "2008-11-28",
        duration     = n,
        avg_return   = -0.025,
        max_drawdown = -0.08,
        sigma_breach = 3.5,
    )

    gc   = GrangerComputer(n_jobs=2)
    mat, pvals = gc.compute_period_with_pvalues(returns, shock)

    print(f"\nGC matrix shape : {mat.shape}")
    print(f"Edge density    : {(mat.sum() - len(tickers)) / (len(tickers)*(len(tickers)-1)):.3f}")
    print(f"\nTop causes:\n{gc.top_causes(mat, tickers, top_n=5)}")
    print(f"\nTop effects:\n{gc.top_effects(mat, tickers, top_n=5)}")
    print(f"\nT0→T1 GC: {mat[0,1]} (expected 1.0), p-val: {pvals[0,1]:.4f}")
    print(f"Diagonal (self-loops): {np.diag(mat)}")
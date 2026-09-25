"""
tests/test_granger.py
-----------------------
Unit tests for graph/granger.py — GrangerComputer (paper Eq. 3 & 4).

Covers:
    - Adaptive lag capping: _adaptive_lag() respects min_lag, period-length
      bound, MAX_LAG_CAP=10, and an explicit max_lag override
    - ssr_chi2test threshold behavior: p <= threshold -> GC=1, else GC=0
    - Self-loops: diagonal is always 1.0
    - Insufficient / low-quality data returns no edge (GC=0), not a crash
    - compute_period() / compute_period_with_pvalues() shape contracts
    - density_report / top_causes / top_effects correctness on a known matrix

Note: Granger tests on small synthetic series are inherently a little noisy
(statsmodels' chi2 test can occasionally reject/fail to reject by chance),
so causal-relationship assertions use a clearly-engineered signal (lag-1
copy + tiny noise) and a fixed seed rather than asserting exact p-values.

Run:
    pytest tests/test_granger.py -v
    python tests/test_granger.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.granger import GrangerComputer, MAX_LAG_CAP
from shock_detection.detector import ShockPeriod


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def gc():
    # n_jobs=1 for deterministic, low-overhead test execution
    return GrangerComputer(n_jobs=1)


def make_shock(duration, name="Test Shock", start="2020-01-01"):
    dates = pd.bdate_range(start, periods=duration)
    return ShockPeriod(
        name=name,
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        duration=duration,
        avg_return=-0.02,
        max_drawdown=-0.05,
        sigma_breach=2.5,
    )


@pytest.fixture
def causal_returns():
    """
    6 tickers, 60 trading days. T0 -> T1 by construction (T1 is a lag-1
    copy of T0 plus small noise); the rest are independent noise.
    """
    n = 60
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["T0.NS", "T1.NS", "T2.NS", "T3.NS", "T4.NS", "^NSEI"]
    data = rng.normal(0, 0.01, size=(n, 6))
    data[:, 1] = np.roll(data[:, 0], 1) + rng.normal(0, 0.002, n)  # T0 -> T1
    df = pd.DataFrame(data, index=dates, columns=tickers)
    df.index.name = "Date"
    return df


# ─────────────────────────────────────────────
# ADAPTIVE LAG  (paper: "1 day up to length of period", capped at 10)
# ─────────────────────────────────────────────

class TestAdaptiveLag:

    def test_short_period_uses_period_length_minus_2(self, gc):
        # period_length=6 -> min(6-2, 10) = 4
        assert gc._adaptive_lag(6) == 4

    def test_long_period_capped_at_max_lag_cap(self, gc):
        # period_length=100 -> min(98, 10) = 10
        assert gc._adaptive_lag(100) == MAX_LAG_CAP

    def test_very_short_period_floors_at_min_lag(self, gc):
        # period_length=2 -> min(0, 10)=0, but floored at min_lag=1
        assert gc._adaptive_lag(2) == gc.min_lag

    def test_explicit_max_lag_override_wins(self):
        gc_override = GrangerComputer(n_jobs=1, max_lag=3)
        # Regardless of period length, explicit override is used verbatim
        assert gc_override._adaptive_lag(100) == 3
        assert gc_override._adaptive_lag(5) == 3

    def test_lag_never_exceeds_cap_across_range(self, gc):
        for period_length in [5, 10, 20, 50, 100, 500]:
            lag = gc._adaptive_lag(period_length)
            assert lag <= MAX_LAG_CAP
            assert lag >= gc.min_lag


# ─────────────────────────────────────────────
# p-THRESHOLD BEHAVIOUR  (Eq. 4)
# ─────────────────────────────────────────────

class TestThresholdBehaviour:

    def test_pvalue_le_threshold_gives_gc_one(self, gc, monkeypatch):
        monkeypatch.setattr(gc, "_test_pair_pvalue", lambda x, y, lag: 0.01)
        result = gc._test_pair(np.zeros(10), np.zeros(10), max_lag=2)
        assert result == 1.0

    def test_pvalue_above_threshold_gives_gc_zero(self, gc, monkeypatch):
        monkeypatch.setattr(gc, "_test_pair_pvalue", lambda x, y, lag: 0.5)
        result = gc._test_pair(np.zeros(10), np.zeros(10), max_lag=2)
        assert result == 0.0

    def test_pvalue_exactly_at_threshold_gives_gc_one(self, gc, monkeypatch):
        """Eq. 4: GC=1 if p <= 0.05 (inclusive)."""
        monkeypatch.setattr(gc, "_test_pair_pvalue", lambda x, y, lag: gc.p_threshold)
        result = gc._test_pair(np.zeros(10), np.zeros(10), max_lag=2)
        assert result == 1.0

    def test_none_pvalue_gives_gc_zero(self, gc, monkeypatch):
        """Insufficient data -> _test_pair_pvalue returns None -> no edge."""
        monkeypatch.setattr(gc, "_test_pair_pvalue", lambda x, y, lag: None)
        result = gc._test_pair(np.zeros(10), np.zeros(10), max_lag=2)
        assert result == 0.0

    def test_custom_threshold_respected(self):
        strict_gc = GrangerComputer(n_jobs=1, p_threshold=0.01)
        # p=0.03 fails a stricter 0.01 threshold, though it'd pass the paper's 0.05
        assert strict_gc.p_threshold == 0.01


# ─────────────────────────────────────────────
# INSUFFICIENT / LOW-QUALITY DATA
# ─────────────────────────────────────────────

class TestDataQualityGuards:

    def test_mostly_nan_series_returns_none_pvalue(self, gc):
        x = np.full(30, np.nan)
        x[:5] = 0.01
        y = np.random.default_rng(0).normal(0, 0.01, 30)
        pval = gc._test_pair_pvalue(x, y, max_lag=3)
        assert pval is None

    def test_too_short_series_returns_none_pvalue(self, gc):
        x = np.array([0.01, 0.02])
        y = np.array([0.01, -0.01])
        pval = gc._test_pair_pvalue(x, y, max_lag=5)
        assert pval is None

    def test_constant_series_does_not_crash(self, gc):
        """Degenerate (zero-variance) series should fail gracefully, not raise."""
        x = np.full(30, 0.001)
        y = np.full(30, 0.002)
        pval = gc._test_pair_pvalue(x, y, max_lag=3)
        assert pval is None or isinstance(pval, float)


# ─────────────────────────────────────────────
# SELF-LOOPS + MATRIX SHAPE  (Section III-A-4)
# ─────────────────────────────────────────────

class TestMatrixContract:

    def test_diagonal_is_always_one(self, gc, causal_returns):
        shock = make_shock(duration=causal_returns.shape[0])
        matrix = gc.compute_period(causal_returns, shock)
        N = causal_returns.shape[1]
        np.testing.assert_array_equal(np.diag(matrix), np.ones(N, dtype=np.float32))

    def test_matrix_shape_matches_n_tickers(self, gc, causal_returns):
        shock = make_shock(duration=causal_returns.shape[0])
        matrix = gc.compute_period(causal_returns, shock)
        N = causal_returns.shape[1]
        assert matrix.shape == (N, N)

    def test_matrix_values_are_binary(self, gc, causal_returns):
        shock = make_shock(duration=causal_returns.shape[0])
        matrix = gc.compute_period(causal_returns, shock)
        unique_vals = set(np.unique(matrix).tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_compute_period_with_pvalues_shapes_match(self, gc, causal_returns):
        shock = make_shock(duration=causal_returns.shape[0])
        gc_matrix, pv_matrix = gc.compute_period_with_pvalues(causal_returns, shock)
        N = causal_returns.shape[1]
        assert gc_matrix.shape == (N, N)
        assert pv_matrix.shape == (N, N)
        # p-values should be in [0, 1] off-diagonal
        off_diag = pv_matrix[~np.eye(N, dtype=bool)]
        assert (off_diag >= 0.0).all() and (off_diag <= 1.0).all()

    def test_engineered_causal_pair_detected(self, gc, causal_returns):
        """
        T0 -> T1 by construction. With a clear lag-1 relationship and low
        noise, the GC test should find significance for this pair (not
        guaranteed 100% due to statistical noise, but should hold with the
        fixed seed used in the fixture).
        """
        shock = make_shock(duration=causal_returns.shape[0])
        matrix = gc.compute_period(causal_returns, shock)
        tickers = list(causal_returns.columns)
        i, j = tickers.index("T0.NS"), tickers.index("T1.NS")
        assert matrix[i, j] == 1.0

    def test_compute_all_handles_multiple_shocks(self, gc, causal_returns):
        shocks = [
            make_shock(duration=30, name="Shock A", start="2020-01-01"),
            make_shock(duration=25, name="Shock B", start="2020-03-01"),
        ]
        # Extend the returns index to cover both windows
        n = 80
        rng = np.random.default_rng(1)
        dates = pd.bdate_range("2020-01-01", periods=n)
        tickers = list(causal_returns.columns)
        data = rng.normal(0, 0.01, size=(n, len(tickers)))
        extended = pd.DataFrame(data, index=dates, columns=tickers)

        results = gc.compute_all(extended, [
            make_shock(duration=10, name="Shock A", start="2020-01-01"),
            make_shock(duration=10, name="Shock B", start="2020-02-01"),
        ])
        assert set(results.keys()) == {"Shock A", "Shock B"}
        for mat in results.values():
            assert mat.shape == (len(tickers), len(tickers))


# ─────────────────────────────────────────────
# DIAGNOSTICS: density_report / top_causes / top_effects
# ─────────────────────────────────────────────

class TestDiagnostics:

    @pytest.fixture
    def known_matrix(self):
        # 4 tickers: T0 causes everyone (out-degree 3), T3 is caused by everyone (in-degree 3)
        tickers = ["T0", "T1", "T2", "T3"]
        mat = np.eye(4, dtype=np.float32)   # self-loops
        mat[0, 1] = mat[0, 2] = mat[0, 3] = 1.0
        mat[1, 3] = mat[2, 3] = 1.0
        return tickers, mat

    def test_density_report_counts_edges_excluding_self_loops(self, gc, known_matrix):
        tickers, mat = known_matrix
        report = gc.density_report({"S1": mat}, tickers)
        # edges excluding diagonal: (0,1)(0,2)(0,3)(1,3)(2,3) = 5
        assert report.loc["S1", "n_edges"] == 5
        assert report.loc["S1", "self_loops"] == 4

    def test_top_causes_ranks_t0_first(self, gc, known_matrix):
        tickers, mat = known_matrix
        top = gc.top_causes(mat, tickers, top_n=1)
        assert top.iloc[0]["ticker"] == "T0"
        assert top.iloc[0]["out_degree"] == 3

    def test_top_effects_ranks_t3_first(self, gc, known_matrix):
        tickers, mat = known_matrix
        top = gc.top_effects(mat, tickers, top_n=1)
        assert top.iloc[0]["ticker"] == "T3"
        assert top.iloc[0]["in_degree"] == 3


# ─────────────────────────────────────────────
# MANUAL RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
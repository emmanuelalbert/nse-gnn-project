"""
tests/test_returns.py
----------------------
Unit tests for preprocessing/returns.py — ReturnCalculator (paper Eq. 1).

Covers:
    - Core log-return formula correctness against hand-computed values
    - Shape contract: (T, N) prices -> (T-1, N) returns, first row dropped
    - Zero-price handling (-> NaN, not -inf)
    - ±inf sanitisation
    - Validation errors: empty df, single row, non-DatetimeIndex
    - long_term_stats() mean/std correctness
    - period_mean_return / period_volatility / amplitude (Eq. 20) /
      average_change (Eq. 19) over a sliced window

Run:
    pytest tests/test_returns.py -v
    python tests/test_returns.py            # quick manual run
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.returns import ReturnCalculator


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def calc():
    return ReturnCalculator()


@pytest.fixture
def simple_prices():
    """3 tickers, 5 trading days, no missing data."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "A.NS": [100.0, 110.0, 121.0, 108.9, 108.9],
            "B.NS": [50.0, 50.0, 55.0, 55.0, 60.5],
            "^NSEI": [10000.0, 10100.0, 10201.0, 10099.0, 10200.0],
        },
        index=dates,
    )
    prices.index.name = "Date"
    return prices


# ─────────────────────────────────────────────
# CORE FORMULA CORRECTNESS  (Eq. 1)
# ─────────────────────────────────────────────

class TestComputeCorrectness:

    def test_known_values(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        # A.NS: 100 -> 110 -> 121 -> 108.9 -> 108.9
        expected_A = [
            np.log(110.0 / 100.0),
            np.log(121.0 / 110.0),
            np.log(108.9 / 121.0),
            np.log(108.9 / 108.9),   # = 0.0 (flat day)
        ]
        np.testing.assert_allclose(returns["A.NS"].values, expected_A, atol=1e-10)

    def test_flat_price_gives_zero_return(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        # Last day of A.NS is flat (108.9 -> 108.9)
        assert returns["A.NS"].iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_shape_contract(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        T, N = simple_prices.shape
        assert returns.shape == (T - 1, N)

    def test_first_row_dropped_not_nan_leaked(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        # No row should correspond to the very first price date
        assert simple_prices.index[0] not in returns.index
        assert returns.index[0] == simple_prices.index[1]

    def test_columns_preserved(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        assert list(returns.columns) == list(simple_prices.columns)


# ─────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_price_becomes_nan_not_inf(self, calc):
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        prices = pd.DataFrame(
            {"A.NS": [100.0, 0.0, 105.0, 106.0]},
            index=dates,
        )
        returns = calc.compute(prices)
        # log(0/100) and log(105/0) would both be non-finite; must be NaN
        assert not np.isinf(returns.values).any()
        assert returns["A.NS"].isna().sum() >= 1

    def test_empty_dataframe_raises(self, calc):
        empty = pd.DataFrame()
        with pytest.raises(ValueError):
            calc.compute(empty)

    def test_single_row_raises(self, calc):
        dates = pd.date_range("2020-01-01", periods=1, freq="B")
        prices = pd.DataFrame({"A.NS": [100.0]}, index=dates)
        with pytest.raises(ValueError):
            calc.compute(prices)

    def test_non_datetime_index_raises(self, calc):
        prices = pd.DataFrame(
            {"A.NS": [100.0, 101.0, 102.0]},
            index=[0, 1, 2],
        )
        with pytest.raises(TypeError):
            calc.compute(prices)

    def test_two_rows_minimum_succeeds(self, calc):
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        prices = pd.DataFrame({"A.NS": [100.0, 105.0]}, index=dates)
        returns = calc.compute(prices)
        assert returns.shape == (1, 1)
        assert returns["A.NS"].iloc[0] == pytest.approx(np.log(105.0 / 100.0))

    def test_negative_price_does_not_crash(self, calc):
        # Data errors from yfinance shouldn't be negative, but guard anyway:
        # log of negative -> NaN (numpy warns, doesn't raise), pipeline must survive.
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"A.NS": [100.0, -5.0, 102.0]}, index=dates)
        with np.errstate(invalid="ignore"):
            returns = calc.compute(prices)
        assert returns.shape == (2, 1)


# ─────────────────────────────────────────────
# LONG-TERM STATS
# ─────────────────────────────────────────────

class TestLongTermStats:

    def test_mean_and_std_match_manual_calc(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        stats = calc.long_term_stats(returns)
        assert stats.loc["A.NS", "mean"] == pytest.approx(returns["A.NS"].mean())
        assert stats.loc["A.NS", "std"] == pytest.approx(returns["A.NS"].std(ddof=1))

    def test_stats_columns(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        stats = calc.long_term_stats(returns)
        assert list(stats.columns) == ["mean", "std"]
        assert set(stats.index) == set(simple_prices.columns)


# ─────────────────────────────────────────────
# PERIOD-SLICED METRICS  (mean, volatility, amplitude Eq.20, avg change Eq.19)
# ─────────────────────────────────────────────

class TestPeriodMetrics:

    def test_period_mean_return_matches_slice(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        start, end = returns.index[0], returns.index[-1]
        mean_ret = calc.period_mean_return(returns, str(start.date()), str(end.date()))
        assert mean_ret["A.NS"] == pytest.approx(returns["A.NS"].mean())

    def test_period_mean_return_empty_raises(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        with pytest.raises(ValueError):
            calc.period_mean_return(returns, "2099-01-01", "2099-01-05")

    def test_amplitude_eq20(self, calc, simple_prices):
        """Eq. 20: Amplitude = MaxDailyChange - MinDailyChange (on log returns)."""
        returns = calc.compute(simple_prices)
        start, end = returns.index[0], returns.index[-1]
        amp = calc.amplitude(returns, str(start.date()), str(end.date()))
        expected = returns["A.NS"].max() - returns["A.NS"].min()
        assert amp["A.NS"] == pytest.approx(expected)

    def test_average_change_eq19_uses_raw_prices(self, calc, simple_prices):
        """Eq. 19: AverageChange = mean(P_current - P_previous), on PRICES not returns."""
        start, end = simple_prices.index[0], simple_prices.index[-1]
        avg_change = calc.average_change(simple_prices, str(start.date()), str(end.date()))
        diffs = simple_prices["A.NS"].diff().iloc[1:]
        assert avg_change["A.NS"] == pytest.approx(diffs.mean())

    def test_period_volatility_annualised_scales_by_sqrt252(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        start, end = returns.index[0], returns.index[-1]
        raw = calc.period_volatility(returns, str(start.date()), str(end.date()), annualise=False)
        ann = calc.period_volatility(returns, str(start.date()), str(end.date()), annualise=True)
        assert ann["A.NS"] == pytest.approx(raw["A.NS"] * np.sqrt(252))


# ─────────────────────────────────────────────
# DISTRIBUTION REPORT (sanity, used in EDA notebook)
# ─────────────────────────────────────────────

class TestDistributionReport:

    def test_report_has_expected_columns(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        report = calc.distribution_report(returns)
        expected_cols = {"mean", "std", "skew", "kurtosis", "min", "max", "missing_pct"}
        assert expected_cols.issubset(set(report.columns))

    def test_report_missing_pct_zero_when_no_nans(self, calc, simple_prices):
        returns = calc.compute(simple_prices)
        report = calc.distribution_report(returns)
        assert (report["missing_pct"] == 0.0).all()


# ─────────────────────────────────────────────
# MANUAL RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
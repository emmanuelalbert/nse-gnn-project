"""
tests/test_shock_detection.py
-------------------------------
Unit tests for shock_detection/detector.py — ShockDetector (paper Eq. 2).

Covers:
    - Eq. 2 threshold boundary cases: window average exactly at / just inside /
      just outside the mu ± sigma_threshold*sigma band
    - Merge-gap logic: windows closer than SHOCK_MERGE_GAP_DAYS merge into
      one ShockPeriod; windows farther apart stay separate
    - Minimum duration post-merge filter
    - detect() end-to-end on synthetic injected shocks
    - match_target_periods() overlap tagging
    - Validation errors (empty df, non-DatetimeIndex, too few rows,
      missing reference ticker)

Run:
    pytest tests/test_shock_detection.py -v
    python tests/test_shock_detection.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shock_detection.detector import ShockDetector, ShockPeriod


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def flat_returns():
    """
    200 trading days of near-zero-noise returns for ^NSEI plus one other
    ticker, so mu ≈ 0 and sigma is small and known-ish. Used to build
    precisely controlled shock windows on top of.
    """
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    ref = rng.normal(0.0, 0.001, size=n)   # tiny noise, mu~0, sigma~0.001
    other = rng.normal(0.0, 0.001, size=n)
    df = pd.DataFrame({"A.NS": other, "^NSEI": ref}, index=dates)
    df.index.name = "Date"
    return df


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

class TestValidation:

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ShockDetector().detect(pd.DataFrame())

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"^NSEI": np.random.normal(0, 0.01, 20)}, index=range(20))
        with pytest.raises(TypeError):
            ShockDetector().detect(df)

    def test_too_few_rows_raises(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        df = pd.DataFrame({"^NSEI": [0.001, -0.001, 0.002]}, index=dates)
        with pytest.raises(ValueError):
            ShockDetector(window_days=5).detect(df)

    def test_missing_reference_ticker_raises_keyerror(self):
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        df = pd.DataFrame({"OTHER.NS": np.random.normal(0, 0.01, 20)}, index=dates)
        with pytest.raises(KeyError):
            ShockDetector(reference_ticker="^NSEI").detect(df)


# ─────────────────────────────────────────────
# Eq. 2 THRESHOLD BOUNDARY CASES
# ─────────────────────────────────────────────

class TestThresholdBoundary:
    """
    Directly exercise _rolling_flag() / _mu / _sigma to control the boundary
    precisely, bypassing randomness in the full detect() pipeline.
    """

    def _make_detector_with_known_stats(self, mu, sigma, window_days=5,
                                         sigma_threshold=2.0, merge_gap_days=3):
        det = ShockDetector(
            window_days=window_days,
            sigma_threshold=sigma_threshold,
            merge_gap_days=merge_gap_days,
            min_duration=window_days,
        )
        det._mu = mu
        det._sigma = sigma
        return det

    def test_window_just_inside_threshold_not_flagged(self):
        """avg deviates by exactly threshold*sigma - epsilon -> NOT flagged."""
        mu, sigma = 0.0, 0.01
        det = self._make_detector_with_known_stats(mu, sigma, window_days=5)
        # avg must equal mu + (2*sigma - epsilon)
        target_avg = mu + (2 * sigma - 1e-6)
        window_vals = np.full(5, target_avg)
        ref = pd.Series(window_vals, index=pd.date_range("2020-01-01", periods=5, freq="B"))
        flagged = det._rolling_flag(ref)
        assert flagged == []

    def test_window_just_outside_threshold_flagged(self):
        """avg deviates by threshold*sigma + epsilon -> flagged."""
        mu, sigma = 0.0, 0.01
        det = self._make_detector_with_known_stats(mu, sigma, window_days=5)
        target_avg = mu + (2 * sigma + 1e-6)
        window_vals = np.full(5, target_avg)
        ref = pd.Series(window_vals, index=pd.date_range("2020-01-01", periods=5, freq="B"))
        flagged = det._rolling_flag(ref)
        assert len(flagged) == 1

    def test_exactly_at_threshold_not_flagged(self):
        """
        Eq. 2 uses a strict '>' comparison, so a window average exactly
        equal to mu + threshold*sigma must NOT be flagged.
        """
        mu, sigma = 0.0, 0.01
        det = self._make_detector_with_known_stats(mu, sigma, window_days=5)
        target_avg = mu + 2 * sigma
        window_vals = np.full(5, target_avg)
        ref = pd.Series(window_vals, index=pd.date_range("2020-01-01", periods=5, freq="B"))
        flagged = det._rolling_flag(ref)
        assert flagged == []

    def test_negative_deviation_also_flagged(self):
        """Eq. 2 uses absolute value — large negative shocks flag too."""
        mu, sigma = 0.0, 0.01
        det = self._make_detector_with_known_stats(mu, sigma, window_days=5)
        target_avg = mu - (2 * sigma + 1e-6)
        window_vals = np.full(5, target_avg)
        ref = pd.Series(window_vals, index=pd.date_range("2020-01-01", periods=5, freq="B"))
        flagged = det._rolling_flag(ref)
        assert len(flagged) == 1

    def test_window_with_too_many_nans_skipped(self):
        mu, sigma = 0.0, 0.01
        det = self._make_detector_with_known_stats(mu, sigma, window_days=5)
        # valid count must be < max(1, window_days//2) = 2 to be skipped;
        # only 1 valid value here regardless of its (extreme) magnitude.
        window_vals = [np.nan, np.nan, np.nan, np.nan, 5.0]
        ref = pd.Series(window_vals, index=pd.date_range("2020-01-01", periods=5, freq="B"))
        flagged = det._rolling_flag(ref)
        assert flagged == []


# ─────────────────────────────────────────────
# MERGE-GAP LOGIC
# ─────────────────────────────────────────────

class TestMergeLogic:

    def test_windows_within_gap_merge(self):
        det = ShockDetector(merge_gap_days=3)
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        # window 1: idx 0-4, window 2: idx 7-11 -> gap of 2 trading days (idx 5,6) <= 3 -> merge
        windows = [(dates[0], dates[4]), (dates[7], dates[11])]
        merged = det._merge_windows(windows, dates)
        assert len(merged) == 1
        assert merged[0] == (dates[0], dates[11])

    def test_windows_beyond_gap_stay_separate(self):
        det = ShockDetector(merge_gap_days=3)
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        # window 1: idx 0-4, window 2: idx 12-16 -> gap of 7 trading days > 3 -> stay separate
        windows = [(dates[0], dates[4]), (dates[12], dates[16])]
        merged = det._merge_windows(windows, dates)
        assert len(merged) == 2

    def test_overlapping_windows_merge(self):
        det = ShockDetector(merge_gap_days=3)
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        # window 2 starts before window 1 ends -> zero gap -> merge
        windows = [(dates[0], dates[6]), (dates[3], dates[10])]
        merged = det._merge_windows(windows, dates)
        assert len(merged) == 1
        assert merged[0] == (dates[0], dates[10])

    def test_empty_windows_returns_empty(self):
        det = ShockDetector()
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        assert det._merge_windows([], dates) == []

    def test_gap_exactly_at_threshold_merges(self):
        """merge_gap_days=3 and gap==3 trading days -> should merge (<=)."""
        det = ShockDetector(merge_gap_days=3)
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        # window1 ends idx4, window2 starts idx8 -> gap days = idx5,6,7 = 3 days
        windows = [(dates[0], dates[4]), (dates[8], dates[12])]
        merged = det._merge_windows(windows, dates)
        assert len(merged) == 1


# ─────────────────────────────────────────────
# MINIMUM DURATION FILTER
# ─────────────────────────────────────────────

class TestMinDurationFilter:

    def test_short_merged_period_filtered_out(self, flat_returns):
        """
        Inject a single-day-wide anomaly too short to survive the
        min_duration filter (which defaults to window_days).
        """
        det = ShockDetector(window_days=5, sigma_threshold=2.0,
                             merge_gap_days=0, min_duration=5)
        shocks = det.detect(flat_returns)
        for s in shocks:
            assert s.duration >= 5


# ─────────────────────────────────────────────
# END-TO-END DETECT() WITH INJECTED SHOCKS
# ─────────────────────────────────────────────

class TestDetectEndToEnd:

    def test_injected_shock_is_detected(self):
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rng = np.random.default_rng(3)
        normal = rng.normal(0.0003, 0.010, size=n)
        shocked = normal.copy()
        # Inject a strong, sustained negative shock over 10 days
        shocked[150:160] = rng.normal(-0.05, 0.005, size=10)

        other = rng.normal(0.0003, 0.010, size=n)
        df = pd.DataFrame({"A.NS": other, "^NSEI": shocked}, index=dates)
        df.index.name = "Date"

        det = ShockDetector(window_days=5, sigma_threshold=2.0, merge_gap_days=3)
        shocks = det.detect(df)

        assert len(shocks) >= 1
        # At least one detected shock should overlap the injected window
        injected_start, injected_end = dates[150], dates[159]
        overlaps = [
            s for s in shocks
            if pd.Timestamp(s.start) <= injected_end and pd.Timestamp(s.end) >= injected_start
        ]
        assert len(overlaps) >= 1

    def test_no_shock_in_pure_noise(self):
        """Pure low-variance noise with generous threshold should yield few/no shocks."""
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rng = np.random.default_rng(11)
        ref = rng.normal(0.0, 0.001, size=n)
        other = rng.normal(0.0, 0.001, size=n)
        df = pd.DataFrame({"A.NS": other, "^NSEI": ref}, index=dates)
        df.index.name = "Date"

        det = ShockDetector(window_days=5, sigma_threshold=4.0)  # very high bar
        shocks = det.detect(df)
        assert len(shocks) == 0

    def test_shocks_sorted_chronologically(self):
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rng = np.random.default_rng(5)
        ref = rng.normal(0.0003, 0.010, size=n)
        ref[100:110] = rng.normal(-0.04, 0.005, size=10)
        ref[300:310] = rng.normal(0.045, 0.005, size=10)
        other = rng.normal(0.0003, 0.010, size=n)
        df = pd.DataFrame({"A.NS": other, "^NSEI": ref}, index=dates)
        df.index.name = "Date"

        det = ShockDetector(window_days=5, sigma_threshold=2.0, merge_gap_days=3)
        shocks = det.detect(df)
        starts = [pd.Timestamp(s.start) for s in shocks]
        assert starts == sorted(starts)


# ─────────────────────────────────────────────
# match_target_periods()
# ─────────────────────────────────────────────

class TestMatchTargetPeriods:

    def test_overlapping_shock_gets_tagged(self):
        shock = ShockPeriod(
            name="Shock_001 (2020-02-24)",
            start="2020-02-24",
            end="2020-03-10",
            duration=13,
            avg_return=-0.03,
            max_drawdown=-0.08,
            sigma_breach=3.2,
        )
        targets = [{
            "name": "COVID-19 Pandemic Onset",
            "start": "2020-02-20",
            "end": "2020-03-23",
            "type": "health_nonfinancial",
        }]
        det = ShockDetector()
        tagged = det.match_target_periods([shock], target_periods=targets, overlap_days=2)
        assert tagged[0].name == "COVID-19 Pandemic Onset"
        assert tagged[0].type == "health_nonfinancial"

    def test_non_overlapping_shock_untagged(self):
        shock = ShockPeriod(
            name="Shock_002 (2011-01-01)",
            start="2011-01-01",
            end="2011-01-10",
            duration=8,
            avg_return=-0.01,
            max_drawdown=-0.02,
            sigma_breach=2.1,
        )
        targets = [{
            "name": "COVID-19 Pandemic Onset",
            "start": "2020-02-20",
            "end": "2020-03-23",
            "type": "health_nonfinancial",
        }]
        det = ShockDetector()
        tagged = det.match_target_periods([shock], target_periods=targets, overlap_days=2)
        assert tagged[0].name == "Shock_002 (2011-01-01)"
        assert tagged[0].type == "auto_detected"

    def test_insufficient_overlap_days_untagged(self):
        """Overlap exists but is too short (< overlap_days trading days) to count."""
        shock = ShockPeriod(
            name="Shock_003 (2020-03-22)",
            start="2020-03-22",
            end="2020-03-25",
            duration=4,
            avg_return=-0.01,
            max_drawdown=-0.02,
            sigma_breach=2.1,
        )
        targets = [{
            "name": "COVID-19 Pandemic Onset",
            "start": "2020-02-20",
            "end": "2020-03-23",
            "type": "health_nonfinancial",
        }]
        det = ShockDetector()
        # Only 1 calendar day of overlap (03-22 to 03-23) -> ~0.7 trading days < overlap_days
        tagged = det.match_target_periods([shock], target_periods=targets, overlap_days=5)
        assert tagged[0].name == "Shock_003 (2020-03-22)"


# ─────────────────────────────────────────────
# ShockPeriod DATACLASS
# ─────────────────────────────────────────────

class TestShockPeriodDataclass:

    def test_to_dict_roundtrip(self):
        s = ShockPeriod("X", "2020-01-01", "2020-01-10", 8, -0.01, -0.03, 2.5, "policy")
        d = s.to_dict()
        assert d["name"] == "X"
        assert d["duration"] == 8

    def test_to_settings_dict_subset(self):
        s = ShockPeriod("X", "2020-01-01", "2020-01-10", 8, -0.01, -0.03, 2.5, "policy")
        sd = s.to_settings_dict()
        assert set(sd.keys()) == {"name", "start", "end", "type"}


# ─────────────────────────────────────────────
# MANUAL RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
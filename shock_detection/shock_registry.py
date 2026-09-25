"""
shock_detection/shock_registry.py
----------------------------------
Persistent store for all detected shock periods, with full CRUD operations,
JSON serialisation, and label-assignment utilities.

Role in the pipeline:
    1. ShockDetector.detect() → raw List[ShockPeriod]
    2. ShockRegistry.save()   → persisted to results/metrics/shock_registry.json
    3. ShockRegistry.load()   → re-loaded by graph/granger.py, training/, analysis/

Why a registry (not just re-running the detector each time)?
  - Granger causality computation is expensive (~2,601 pairs × N shocks).
    Persisting the shock list means we only run it once and reuse it.
  - Manual overrides: the auto-detector may split or miss a well-known crisis.
    The registry lets us add, remove, or relabel periods without touching the
    detector logic.
  - Audit trail: the registry records when detection was run, what parameters
    were used, and which shocks were manually labelled — important for
    reproducibility in a research paper.

The registry saves two files:
    results/metrics/shock_registry.json   — machine-readable, versioned
    results/metrics/shock_registry.csv    — human-readable summary table

Usage:
    from shock_detection.shock_registry import ShockRegistry
    from shock_detection.detector import ShockPeriod

    registry = ShockRegistry()
    registry.save(shocks)                          # persist detected shocks
    shocks   = registry.load()                     # reload as ShockPeriod list
    registry.label(name="Shock_005 (2008-10-06)",
                   new_name="Global Financial Crisis",
                   new_type="global_financial")    # manual override
    df       = registry.to_dataframe()             # for display / analysis
    target   = registry.get_target_periods()       # only the 4 key shocks
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as CFG
from shock_detection.detector import ShockPeriod

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

REGISTRY_JSON = CFG.METRICS_DIR / "shock_registry.json"
REGISTRY_CSV  = CFG.METRICS_DIR / "shock_registry.csv"


# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────

class ShockRegistry:
    """
    Persistent, version-tracked store for ShockPeriod objects.

    The underlying JSON file has the following structure:
    {
        "metadata": {
            "created_at"       : "2025-06-14T10:00:00",
            "last_updated"     : "2025-06-14T10:00:00",
            "n_shocks"         : 47,
            "detector_params"  : { "sigma_threshold": 2.0, "window_days": 5, ... },
            "data_range"       : { "start": "2005-01-03", "end": "2025-05-30" }
        },
        "shocks": [ { ...ShockPeriod fields... }, ... ]
    }
    """

    def __init__(self, path: Path = REGISTRY_JSON):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: Dict = {}

    # ─────────────────────────────────────────
    # PERSIST
    # ─────────────────────────────────────────

    def save(
        self,
        shocks: List[ShockPeriod],
        detector_params: Optional[dict] = None,
        data_range: Optional[dict] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Persist a list of ShockPeriod objects to JSON.

        Args:
            shocks          : Detected (or manually curated) shock list.
            detector_params : Dict of detector hyperparameters for audit trail.
            data_range      : {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}.
            overwrite       : If False and file exists, raises FileExistsError.
        """
        if self.path.exists() and not overwrite:
            raise FileExistsError(
                f"Registry already exists at {self.path}. "
                "Pass overwrite=True to replace it."
            )

        now = datetime.now().isoformat(timespec="seconds")

        payload = {
            "metadata": {
                "created_at"      : now,
                "last_updated"    : now,
                "n_shocks"        : len(shocks),
                "detector_params" : detector_params or self._default_params(),
                "data_range"      : data_range or {},
            },
            "shocks": [s.to_dict() for s in shocks],
        }

        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        self._registry = payload
        logger.info(f"Registry saved: {len(shocks)} shocks → {self.path}")

        # Also write human-readable CSV
        self._save_csv(shocks)

    def load(self) -> List[ShockPeriod]:
        """
        Load shock periods from the JSON registry.

        Returns:
            List of ShockPeriod objects, sorted chronologically.

        Raises:
            FileNotFoundError: if the registry has not been saved yet.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"No shock registry found at {self.path}. "
                "Run ShockDetector.detect() and ShockRegistry.save() first."
            )

        with open(self.path) as f:
            payload = json.load(f)

        self._registry = payload
        shocks = [ShockPeriod(**s) for s in payload["shocks"]]
        shocks = sorted(shocks, key=lambda s: s.start)

        logger.info(
            f"Registry loaded: {len(shocks)} shocks from {self.path} "
            f"(last updated: {payload['metadata'].get('last_updated', 'unknown')})"
        )
        return shocks

    def exists(self) -> bool:
        """Return True if the registry file exists on disk."""
        return self.path.exists()

    # ─────────────────────────────────────────
    # CRUD
    # ─────────────────────────────────────────

    def label(
        self,
        name: str,
        new_name: Optional[str] = None,
        new_type: Optional[str] = None,
    ) -> None:
        """
        Override the name and/or type of a shock period by its current name.
        Writes the updated registry back to disk.

        Args:
            name     : Current name of the shock to update.
            new_name : Replacement name (e.g. "Global Financial Crisis").
            new_type : Replacement type (e.g. "global_financial").
        """
        shocks  = self.load()
        updated = False

        for shock in shocks:
            if shock.name == name:
                if new_name:
                    shock.name = new_name
                if new_type:
                    shock.type = new_type
                updated = True
                logger.info(
                    f"Label updated: '{name}' → name='{shock.name}', type='{shock.type}'"
                )
                break

        if not updated:
            raise KeyError(
                f"Shock '{name}' not found in registry. "
                f"Available names: {[s.name for s in shocks[:5]]} ..."
            )

        self._update_timestamp()
        self.save(shocks, overwrite=True)

    def add(self, shock: ShockPeriod) -> None:
        """
        Manually add a shock period that was not auto-detected.
        Useful for injecting known events that fell below the 2σ threshold
        (e.g. a sharp 3-day crash that didn't meet the 5-day minimum).
        """
        shocks = self.load() if self.exists() else []

        # Check for overlap with existing shocks
        for existing in shocks:
            if self._overlaps(shock, existing):
                logger.warning(
                    f"New shock '{shock.name}' [{shock.start}→{shock.end}] "
                    f"overlaps with existing '{existing.name}' "
                    f"[{existing.start}→{existing.end}]. Adding anyway."
                )

        shocks.append(shock)
        shocks = sorted(shocks, key=lambda s: s.start)
        self.save(shocks, overwrite=True)
        logger.info(f"Added shock: '{shock.name}' [{shock.start} → {shock.end}]")

    def remove(self, name: str) -> None:
        """Remove a shock period by name."""
        shocks  = self.load()
        before  = len(shocks)
        shocks  = [s for s in shocks if s.name != name]

        if len(shocks) == before:
            raise KeyError(f"Shock '{name}' not found in registry.")

        self.save(shocks, overwrite=True)
        logger.info(f"Removed shock: '{name}'")

    def get(self, name: str) -> ShockPeriod:
        """Retrieve a single ShockPeriod by name."""
        shocks = self.load()
        for s in shocks:
            if s.name == name:
                return s
        raise KeyError(f"Shock '{name}' not found in registry.")

    def get_by_type(self, shock_type: str) -> List[ShockPeriod]:
        """Return all shocks of a given type."""
        return [s for s in self.load() if s.type == shock_type]

    def get_target_periods(self) -> List[ShockPeriod]:
        """
        Return only the four key analysis periods (matched from settings).
        These are the shocks used in the comparative analysis section,
        equivalent to the paper's three chosen periods.
        """
        target_names = {t["name"] for t in CFG.TARGET_SHOCK_PERIODS}
        shocks = self.load()
        matched = [s for s in shocks if s.name in target_names]
        if not matched:
            logger.warning(
                "No target periods found in registry. "
                "Run ShockDetector.match_target_periods() before saving."
            )
        return matched

    def get_period_slice(
        self,
        returns: pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:
        """
        Convenience: load a shock by name and slice the returns DataFrame.

        Args:
            returns : Full log returns DataFrame.
            name    : Shock name in the registry.

        Returns:
            returns sliced to [shock.start, shock.end].
        """
        shock = self.get(name)
        mask  = (returns.index >= pd.Timestamp(shock.start)) & \
                (returns.index <= pd.Timestamp(shock.end))
        return returns.loc[mask]

    # ─────────────────────────────────────────
    # INSPECTION
    # ─────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return all shocks as a summary DataFrame.
        Sorted chronologically; 1-based index.
        """
        shocks = self.load()
        rows   = [s.to_dict() for s in shocks]
        df     = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
        df.index += 1
        return df

    def metadata(self) -> dict:
        """Return the registry metadata dict."""
        if not self._registry:
            if not self.exists():
                return {}
            with open(self.path) as f:
                self._registry = json.load(f)
        return self._registry.get("metadata", {})

    def print_summary(self) -> None:
        """Print a formatted summary of all shock periods to stdout."""
        df   = self.to_dataframe()
        meta = self.metadata()

        print("=" * 70)
        print(f"SHOCK REGISTRY  —  {meta.get('n_shocks', '?')} periods")
        print(f"Last updated: {meta.get('last_updated', 'unknown')}")
        print(f"Data range  : {meta.get('data_range', {})}")
        print("=" * 70)
        print(
            df[["name", "start", "end", "duration", "avg_return",
                "sigma_breach", "type"]]
            .to_string(max_colwidth=35)
        )
        print("=" * 70)

        # Highlight target periods
        target_names = {t["name"] for t in CFG.TARGET_SHOCK_PERIODS}
        found = [s for s in self.load() if s.name in target_names]
        if found:
            print(f"\nTarget analysis periods ({len(found)}/{len(CFG.TARGET_SHOCK_PERIODS)} matched):")
            for s in found:
                print(f"  ✓  {s.name} [{s.start} → {s.end}]  ({s.duration} days, {s.sigma_breach}σ)")
        else:
            print("\n⚠ No target periods matched. Run match_target_periods().")

    def year_distribution(self) -> pd.Series:
        """Count of shock periods per calendar year."""
        shocks = self.load()
        years  = [int(s.start[:4]) for s in shocks]
        return pd.Series(years).value_counts().sort_index()

    def type_distribution(self) -> pd.Series:
        """Count of shock periods per type."""
        shocks = self.load()
        types  = [s.type for s in shocks]
        return pd.Series(types).value_counts()

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _save_csv(self, shocks: List[ShockPeriod]) -> None:
        df = pd.DataFrame([s.to_dict() for s in shocks])
        df.to_csv(REGISTRY_CSV, index=False)
        logger.debug(f"CSV summary saved → {REGISTRY_CSV}")

    def _update_timestamp(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                payload = json.load(f)
            payload["metadata"]["last_updated"] = datetime.now().isoformat(timespec="seconds")
            with open(self.path, "w") as f:
                json.dump(payload, f, indent=2, default=str)

    def _default_params(self) -> dict:
        return {
            "sigma_threshold"  : CFG.SHOCK_SIGMA_THRESHOLD,
            "window_days"      : CFG.SHOCK_WINDOW_DAYS,
            "merge_gap_days"   : CFG.SHOCK_MERGE_GAP_DAYS,
            "reference_ticker" : CFG.SHOCK_REFERENCE_TICKER,
            "granger_p"        : CFG.GRANGER_P_THRESHOLD,
        }

    @staticmethod
    def _overlaps(a: ShockPeriod, b: ShockPeriod) -> bool:
        return (
            pd.Timestamp(a.start) <= pd.Timestamp(b.end) and
            pd.Timestamp(b.start) <= pd.Timestamp(a.end)
        )


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil
    logging.basicConfig(level="DEBUG", format=CFG.LOG_FORMAT)

    # Use a temp dir so the test doesn't pollute results/
    tmp = Path(tempfile.mkdtemp())
    registry = ShockRegistry(path=tmp / "shock_registry.json")

    # Build fake shocks
    shocks = [
        ShockPeriod("Shock_001 (2008-10-06)", "2008-10-06", "2008-10-24",
                    14, -0.031, -0.082, 4.2, "auto_detected"),
        ShockPeriod("Shock_002 (2018-09-21)", "2018-09-21", "2018-10-10",
                    14, -0.018, -0.045, 2.8, "auto_detected"),
        ShockPeriod("Shock_003 (2020-02-24)", "2020-02-24", "2020-03-23",
                    21, -0.038, -0.135, 5.1, "auto_detected"),
    ]

    registry.save(shocks, data_range={"start": "2005-01-03", "end": "2025-05-30"})

    # Reload
    loaded = registry.load()
    assert len(loaded) == 3, f"Expected 3, got {len(loaded)}"

    # Label override
    registry.label("Shock_001 (2008-10-06)", new_name="Global Financial Crisis", new_type="global_financial")
    gfc = registry.get("Global Financial Crisis")
    assert gfc.type == "global_financial"

    # Add a new shock
    registry.add(ShockPeriod("Demonetization Shock", "2016-11-08", "2016-11-25",
                             13, -0.012, -0.030, 2.1, "policy"))
    assert len(registry.load()) == 4

    # Remove
    registry.remove("Demonetization Shock")
    assert len(registry.load()) == 3

    # Summary
    registry.print_summary()

    shutil.rmtree(tmp)
    print("\nAll ShockRegistry tests PASSED.")
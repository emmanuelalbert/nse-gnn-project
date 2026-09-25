from pathlib import Path

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings as CFG

STYLE = {
    "figure.facecolor" : "#ffffff",
    "axes.facecolor"   : "#ffffff",
    "axes.edgecolor"   : "#d0d7de",
    "axes.labelcolor"  : "#1f2328",
    "xtick.color"      : "#57606a",
    "ytick.color"      : "#57606a",
    "text.color"       : "#1f2328",
    "grid.color"       : "#e1e4e8",
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.5,
    "legend.facecolor" : "#ffffff",
    "legend.edgecolor" : "#d0d7de",
    "font.family"      : "DejaVu Sans",
    "font.size"        : 9,
}

# Consistent colour cycle for multi-curve comparison plots
COLOR_CYCLE = [
    "#58a6ff",  # blue
    "#f0883e",  # orange
    "#3fb950",  # green
    "#ff7b72",  # red
    "#a371f7",  # purple
    "#ffa657",  # amber
    "#39c5cf",  # cyan
    "#f778ba",  # pink
]


def apply_style() -> None:
    """Apply the shared dark theme to matplotlib's rcParams."""
    plt.rcParams.update(STYLE)


def save_figure(
    fig,
    filename : str,
    save_dir : Path = None,
    dpi      : int  = 650,
) -> Path:
    """
    Save a figure to the standard figures directory and close it.

    Args:
        fig      : Matplotlib Figure object.
        filename : Output filename (e.g. "roc_curve.png").
        save_dir : Target directory. Defaults to CFG.FIGURES_DIR, read at
                   CALL TIME (not import time) via the None sentinel below.
                   This matters because tests/callers monkeypatch
                   CFG.FIGURES_DIR at runtime (e.g. heatmap.py's
                   __main__ block); a mutable default bound at import
                   time would silently ignore that override.
        dpi      : Resolution.

    Returns:
        Path to the saved file.
    """
    if save_dir is None:
        save_dir = CFG.FIGURES_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
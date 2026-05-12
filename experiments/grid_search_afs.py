"""Grid search over (c, H) for the canonical AFS impact family.

Canonical recursion (per `compute_impact_states_concave`):

    J_t  = (1 - β)·J_{t-1} + q_t / ADV_d
    Ī_t = σ_d · sign(J_t) · |J_t|^c

c=1 reproduces OW on daily-reset; c=0.5 is canonical AFS sqrt.

Output:
    results/afs_grid.csv          — long-format raw rolling-baseline rows
    results/afs_grid_pivot.csv    — (c × H) pivot of median pooled OOS R²
    results/afs_grid_heatmap.png  — visualisation

Caches the bin panel (~2 min build) to saved/_panel_cache.pkl so subsequent
runs start in seconds.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import price_impact as pi  # noqa: E402
from price_impact.fitting import concavity_grid_search  # noqa: E402

DATA_DIR = Path("/Users/AnranSeverac/PriceImpact/data")
CACHE_PATH = ROOT / "saved" / "_panel_cache.pkl"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

C_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
H_GRID = [5, 10, 15, 30, 45, 60, 90, 120, 180]
TAU_BINS = 180
CARRY = "daily"
N_WINDOWS = 10
OFFSET = 2


def load_or_build_panel() -> "pi.PanelData":
    if CACHE_PATH.exists():
        print(f"loading cached panel from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    print(f"building panel from {DATA_DIR} ...")
    t0 = time.time()
    panel = pi.build_panel(DATA_DIR, year=2019, top_n=20)
    print(f"  built in {time.time() - t0:.1f}s — {panel.n_stocks} stocks, {len(panel.bins):,} bins")
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(panel, f)
    print(f"  cached to {CACHE_PATH}")
    return panel


def aggregate(grid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(c, H) summary: median and mean OOS R² across (window × stock)."""
    summary = (
        grid_df.groupby(["c", "H"])
        .agg(
            oos_r2_median=("oos_r2", "median"),
            oos_r2_mean=("oos_r2", "mean"),
            oos_r2_std=("oos_r2", "std"),
            n_obs=("oos_r2", "count"),
            lambda_median=("lambda", "median"),
        )
        .reset_index()
    )
    median_pivot = summary.pivot(index="c", columns="H", values="oos_r2_median")
    return summary, median_pivot


def save_heatmap(median_pivot: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    vals = median_pivot.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(len(median_pivot.index)))
    ax.set_yticklabels([f"{a:.2f}" for a in median_pivot.index])
    ax.set_xticks(range(len(median_pivot.columns)))
    ax.set_xticklabels([f"{int(h)}" for h in median_pivot.columns])
    ax.set_ylabel("Concavity exponent  c")
    ax.set_xlabel("Half-life  H  (min)")
    ax.set_title("Median OOS R² over (c, H) — canonical AFS family")
    plt.colorbar(im, ax=ax, label="Median OOS R²")
    # Annotate cells.
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < np.nanmedian(vals) else "black",
                        fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    panel = load_or_build_panel()
    print(f"panel ready ({time.time() - t0:.1f}s)")

    print(f"\nsweeping c × H = {len(C_GRID)} × {len(H_GRID)} = "
          f"{len(C_GRID) * len(H_GRID)} cells (carry={CARRY}, τ={TAU_BINS} bins)\n")

    t_grid = time.time()
    grid_df = concavity_grid_search(
        panel.bins,
        panel.daily_stats,
        c_grid=C_GRID,
        H_grid_minutes=H_GRID,
        tau_bins=TAU_BINS,
        carry=CARRY,
        n_windows=N_WINDOWS,
        offset=OFFSET,
        progress=True,
    )
    print(f"\ngrid search done in {time.time() - t_grid:.1f}s — {len(grid_df):,} rows")

    raw_path = RESULTS_DIR / "afs_grid.csv"
    grid_df.to_csv(raw_path, index=False)
    print(f"saved {raw_path}")

    summary, median_pivot = aggregate(grid_df)
    summary_path = RESULTS_DIR / "afs_grid_summary.csv"
    pivot_path = RESULTS_DIR / "afs_grid_pivot.csv"
    summary.to_csv(summary_path, index=False)
    median_pivot.to_csv(pivot_path)
    print(f"saved {summary_path}")
    print(f"saved {pivot_path}")

    print("\nMedian OOS R² per (c, H):")
    print(median_pivot.round(5).to_string())

    # argmax
    stacked = median_pivot.stack()
    best_c, best_H = stacked.idxmax()
    best_r2 = float(stacked.max())
    print(f"\n>>> Best (median OOS R²): c* = {best_c:.2f},  H* = {best_H:g} min  →  R² = {best_r2:.5f}")

    # Reference baselines.
    refs = []
    if 1.00 in median_pivot.index and 60 in median_pivot.columns:
        v = float(median_pivot.loc[1.00, 60])
        refs.append(("OW baseline      (c=1.00, H=60)", v))
    if 0.50 in median_pivot.index and 45 in median_pivot.columns:
        v = float(median_pivot.loc[0.50, 45])
        refs.append(("In-code AFS      (c=0.50, H=45)", v))
    if 0.50 in median_pivot.index and 60 in median_pivot.columns:
        v = float(median_pivot.loc[0.50, 60])
        refs.append(("Canonical AFS    (c=0.50, H=60)", v))
    for name, v in refs:
        print(f"  {name}: R² = {v:.5f}   Δ vs best = {best_r2 - v:+.5f}")

    heatmap_path = RESULTS_DIR / "afs_grid_heatmap.png"
    save_heatmap(median_pivot, heatmap_path)
    print(f"\nsaved {heatmap_path}")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

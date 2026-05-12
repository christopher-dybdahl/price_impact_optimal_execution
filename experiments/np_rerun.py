"""Re-run non-parametric extension + γ stress test under canonical AFS.

Models:
    OW:  c=1.00, H=60 min  (linear impact, OW-optimal H from grid)
    AFS: c=0.90, H=45 min  (canonical concavity optimum from grid)

Outputs:
    results/np_headline.csv          — pooled OOS R² (tab:apples)
    results/np_stress.csv            — stress test (tab:stress)
    results/np_summary_ow.csv        — per-window NP summary
    results/np_summary_afs.csv
    figures/impact_curves.png        — NP g(x) curves at train month 3
    figures/stress_tuning.png        — γ tuning curves at 3 window sizes
    figures/parametric_vs_nonparametric.png — bar chart
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

from price_impact.impact_states import compute_impact_states_concave
from price_impact.fitting import (
    build_regression_features,
    daily_sufficient_stats,
    rolling_nonparametric,
    build_bin_stats,
    universal_bin_means,
    regularised_bin_means,
    predict_and_score,
    ols_from_sums,
    STAT_COLS,
)

CACHE = ROOT / "saved" / "_panel_cache.pkl"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

TAU = 180
CARRY = "daily"
NBINS = 15
NWIN = 10

MODELS = {
    "OW":  dict(c=1.00, H=60),
    "AFS": dict(c=0.50, H=45),
}


# ── helpers ────────────────────────────────────────────────────────────────

def load_panel():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def get_features(panel, c, H):
    imp = compute_impact_states_concave(panel.bins, panel.daily_stats, H, c)
    return build_regression_features(imp, panel.bins, TAU, CARRY)


# ── Pooled OOS R² (parametric λ·x) ────────────────────────────────────────

def pooled_param_r2(feats):
    stats = daily_sufficient_stats(feats)
    ss_res_total = 0.0
    g_y = g_yy = g_n = 0.0
    for tm in range(1, NWIN + 1):
        vm = tm + 2
        tr = stats[stats["month"] == tm].groupby("stock")[STAT_COLS].sum()
        va = stats[stats["month"] == vm].groupby("stock")[STAT_COLS].sum()
        ix = tr.index.intersection(va.index)
        if not len(ix):
            continue
        lam, _ = ols_from_sums(tr.loc[ix])
        for s in ix:
            v = va.loc[s]
            ss_res_total += v["yy"] - 2 * lam[s] * v["xy"] + lam[s] ** 2 * v["xx"]
            g_y += v["y"]
            g_yy += v["yy"]
            g_n += v["count"]
    ss_tot = g_yy - g_y ** 2 / g_n if g_n else 0
    return 1 - ss_res_total / ss_tot if ss_tot else float("nan")


# ── Pooled OOS R² (NP regularised) ────────────────────────────────────────

def pooled_np_r2(feats):
    ss_res_total = 0.0
    g_y = g_yy = g_n = 0.0
    for tm in range(1, NWIN + 1):
        te_m, va_m = tm + 1, tm + 2
        tr, edges = build_bin_stats(feats, tm, NBINS)
        te, _ = build_bin_stats(feats, te_m, NBINS, edges)
        va, _ = build_bin_stats(feats, va_m, NBINS, edges)
        if tr.empty or te.empty or va.empty:
            continue
        gb = universal_bin_means(tr)
        mn = float(tr["n"].median())
        gg = mn * np.logspace(-3, 3, 60)
        bg, bm = gg[0], float("inf")
        for g in gg:
            reg = regularised_bin_means(tr, gb, float(g))
            mse, _ = predict_and_score(te, reg)
            if mse < bm:
                bm = mse
                bg = float(g)
        reg_best = regularised_bin_means(tr, gb, bg)
        merged = va.merge(
            reg_best[["stock", "bin", "g_reg"]], on=["stock", "bin"], how="inner"
        )
        ss_res_total += float(
            (merged["syy"] - 2 * merged["g_reg"] * merged["sy"]
             + merged["g_reg"] ** 2 * merged["n"]).sum()
        )
        g_y += merged["sy"].sum()
        g_yy += merged["syy"].sum()
        g_n += merged["n"].sum()
    ss_tot = g_yy - g_y ** 2 / g_n if g_n else 0
    return 1 - ss_res_total / ss_tot if ss_tot else float("nan")


# ── Stress test ────────────────────────────────────────────────────────────

def restrict_days(feats, n_days):
    parts = []
    for m in sorted(feats["month"].unique()):
        mf = feats[feats["month"] == m]
        dates = sorted(mf["date"].unique())
        if len(dates) > n_days:
            dates = dates[-n_days:]
        parts.append(mf[mf["date"].isin(dates)])
    return pd.concat(parts, ignore_index=True)


def stress_row(feats, n_days=None):
    f = restrict_days(feats, n_days) if n_days else feats
    summary, fits = rolling_nonparametric(f, n_bins=NBINS, n_windows=NWIN)
    if summary.empty:
        return None, None
    row = dict(
        median_n=summary["median_n"].median(),
        r2_raw=summary["oos_r2_raw"].median(),
        r2_reg=summary["oos_r2_reg"].median(),
        gain=float((summary["oos_r2_reg"] - summary["oos_r2_raw"]).median()),
        gamma_n=float((summary["best_gamma"] / summary["median_n"]).median()),
    )
    return row, fits


# ── Figures ────────────────────────────────────────────────────────────────

def plot_impact_curves(fits_ow, fits_afs, path, c_ow=1.0, c_afs=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, fits, label in [(axes[0], fits_ow, f"OW (c={c_ow:g})"),
                             (axes[1], fits_afs, f"AFS (c={c_afs:g})")]:
        fit = fits[3]  # train month 3
        tr = fit.train_stats
        gb = fit.g_bar
        edges = fit.bin_edges
        mids = 0.5 * (edges[:-1] + edges[1:])
        mids[0] = edges[1] - (edges[2] - edges[1])
        mids[-1] = edges[-2] + (edges[-2] - edges[-3])

        for stock in tr["stock"].unique():
            sd = tr[tr["stock"] == stock].sort_values("bin")
            ax.plot(mids[sd["bin"].values], sd["mean_y"].values,
                    color="grey", alpha=0.25, lw=0.8)

        gb_arr = gb.reindex(range(NBINS)).values
        ax.plot(mids, gb_arr, "b--", lw=2, label="universal $\\bar g$")
        ax.set_xlabel("$x = \\Delta\\bar I$")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)

    axes[0].set_ylabel("$g(x)$")
    fig.suptitle("Non-parametric $g(x)$ at train month $m=3$", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def plot_stress_tuning(stress_fits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["1 month (headline)", "5 days", "2 days"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, (label, fits) in zip(axes, zip(labels, stress_fits)):
        if fits is None:
            ax.set_title(label)
            continue
        for tm, fit in sorted(fits.items()):
            ax.plot(fit.gamma_grid, fit.gamma_mses, alpha=0.4, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("$\\gamma$")
        ax.set_title(label)

    axes[0].set_ylabel("Test-month MSE")
    fig.suptitle("$\\gamma$ tuning curves — OW at three training-window sizes",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def plot_param_vs_np(headline_df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(headline_df))
    w = 0.35
    ax.bar(x - w / 2, headline_df["parametric"], w, label="Parametric", color="steelblue")
    ax.bar(x + w / 2, headline_df["np_reg"], w, label="NP regularised", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(headline_df["model"])
    ax.set_ylabel("Pooled OOS $R^2$")
    ax.set_title("Parametric vs Non-parametric OOS $R^2$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    panel = load_panel()
    print(f"Panel loaded ({time.time() - t0:.1f}s)\n")

    feats = {}
    for name, cfg in MODELS.items():
        print(f"Computing {name} features (c={cfg['c']}, H={cfg['H']})...", flush=True)
        t = time.time()
        feats[name] = get_features(panel, cfg["c"], cfg["H"])
        print(f"  {len(feats[name]):,} rows in {time.time() - t:.1f}s")

    # ── Headline table (tab:apples) ──
    print("\n=== Headline Pooled OOS R² ===")
    rows = []
    for name in MODELS:
        print(f"  {name}: parametric...", end="", flush=True)
        pr2 = pooled_param_r2(feats[name])
        print(f" {pr2:.4f}  NP...", end="", flush=True)
        nr2 = pooled_np_r2(feats[name])
        print(f" {nr2:.4f}")
        rows.append(dict(model=name, parametric=pr2, np_reg=nr2, delta=nr2 - pr2))
    hdf = pd.DataFrame(rows)
    print("\n" + hdf.to_string(index=False))
    hdf.to_csv(RESULTS / "np_headline.csv", index=False)

    # ── Rolling NP summaries + fits for figures ──
    print("\n=== Rolling NP per-window ===")
    all_fits = {}
    for name in MODELS:
        summary, fits = rolling_nonparametric(feats[name], n_bins=NBINS, n_windows=NWIN)
        all_fits[name] = fits
        print(f"\n{name}:")
        print(summary[["train_month", "best_gamma", "median_n",
                        "oos_r2_raw", "oos_r2_univ", "oos_r2_reg"]].round(5).to_string(index=False))
        summary.to_csv(RESULTS / f"np_summary_{name.lower()}.csv", index=False)

    # ── Stress test ──
    print("\n=== γ Stress Test ===")
    stress = []
    ow_stress_fits = []  # for tuning-curve figure
    for name in MODELS:
        for nd, label in [(None, "1 month"), (5, "5 days"), (2, "2 days")]:
            print(f"  {name} / {label}...", end="", flush=True)
            row, fits = stress_row(feats[name], nd)
            if row:
                row["model"] = name
                row["window"] = label
                stress.append(row)
                print(f" gain={row['gain']:+.4f}, γ/n={row['gamma_n']:.2f}")
                if name == "OW":
                    ow_stress_fits.append(fits)
            else:
                print(" EMPTY")
                if name == "OW":
                    ow_stress_fits.append(None)

    sdf = pd.DataFrame(stress)
    print("\nStress table:")
    print(sdf.to_string(index=False))
    sdf.to_csv(RESULTS / "np_stress.csv", index=False)

    # ── Figures ──
    print("\nGenerating figures...", flush=True)
    if 3 in all_fits["OW"] and 3 in all_fits["AFS"]:
        plot_impact_curves(
            all_fits["OW"], all_fits["AFS"],
            FIGURES / "impact_curves.png",
            c_ow=MODELS["OW"]["c"], c_afs=MODELS["AFS"]["c"],
        )
        print("  saved figures/impact_curves.png")

    if len(ow_stress_fits) == 3:
        plot_stress_tuning(ow_stress_fits,
                           FIGURES / "stress_tuning.png")
        print("  saved figures/stress_tuning.png")

    plot_param_vs_np(hdf, FIGURES / "parametric_vs_nonparametric.png")
    print("  saved figures/parametric_vs_nonparametric.png")

    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

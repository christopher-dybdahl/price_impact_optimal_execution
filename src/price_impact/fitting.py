"""Impact model fitting — parametric (OW / AFS) and non-parametric.

Consumes the precomputed impact-state panel from :mod:`impact_states` and
performs:

  * sufficient-statistics aggregation
  * per-stock OLS (with R^2 / MSE from sums)
  * rolling monthly baseline (train m -> validate m + offset)
  * universal half-life grid search
  * non-parametric binned estimator with cross-stock regularisation
    (the "extended" model from Anran's Section 2.2 write-up)

The carry mode (`carry='daily'` vs `carry='multi'`) selects which Ī column
to use as the regressor, so the same fitting machinery works for both the
daily-reset and the multi-day carry impact states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .impact_states import (
    ModelType,
    compute_impact_states,
    select_i_bar_column,
)

CarryMode = Literal["daily", "multi"]

STAT_COLS = ["xy", "xx", "yy", "x", "y", "count"]


# ---------------------------------------------------------------------------
# Observation-level features.
# ---------------------------------------------------------------------------
def build_regression_features(
    impact_states_df: pd.DataFrame,
    data: pd.DataFrame,
    tau_bins: int,
    carry: CarryMode = "daily",
    start_time: str = "10:00:00",
    price_col: str = "mid",
) -> pd.DataFrame:
    """Build per-bin (x, y) regression features for impact-model OLS.

    x_t = Ī_t - Ī_{t-τ}   (change in normalised impact state over τ bins)
    y_t = (P_t - P_{t-τ}) / P_{t-τ}   (τ-bin return on `price_col`)

    Bins before `start_time` are dropped to avoid open effects.
    """
    i_col = select_i_bar_column(carry)
    df = impact_states_df[["stock", "date", "time", i_col]].merge(
        data[["stock", "date", "time", price_col]],
        on=["stock", "date", "time"],
        how="inner",
    )
    df = df.rename(columns={price_col: "price"})
    df = df.sort_values(["stock", "date", "time"]).reset_index(drop=True)

    df["I_lag"] = df.groupby(["stock", "date"])[i_col].shift(tau_bins)
    df["price_lag"] = df.groupby(["stock", "date"])["price"].shift(tau_bins)

    df["x"] = df[i_col] - df["I_lag"]
    df["y"] = (df["price"] - df["price_lag"]) / df["price_lag"]
    df = df.dropna(subset=["x", "y"])

    if start_time:
        df = df[df["time"] >= start_time]

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    return df[["stock", "date", "month", "time", "x", "y"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sufficient statistics + OLS helpers.
# ---------------------------------------------------------------------------
def daily_sufficient_stats(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(stock, date) sums of x, y, xy, xx, yy, count.

    These sums are additive across days, so they support fast rolling.
    """
    df = features.copy()
    df["xy"] = df["x"] * df["y"]
    df["xx"] = df["x"] ** 2
    df["yy"] = df["y"] ** 2
    df["count"] = 1
    summary = df.groupby(["stock", "date"], sort=False)[STAT_COLS].sum().reset_index()
    summary["month"] = pd.to_datetime(summary["date"]).dt.month
    return summary.loc[summary["yy"] > 1e-12].copy()


def ols_from_sums(s: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (slope λ, intercept α) Series indexed like `s`, from sums."""
    n = s["count"]
    cov_xy = s["xy"] - s["x"] * s["y"] / n
    var_x = s["xx"] - s["x"] ** 2 / n
    lam = cov_xy / var_x
    alpha = s["y"] / n - lam * s["x"] / n
    return lam, alpha


def r2_from_sums(s: pd.DataFrame, lam: pd.Series, alpha: pd.Series) -> pd.Series:
    n = s["count"]
    ss_tot = s["yy"] - s["y"] ** 2 / n
    ss_res = (
        s["yy"]
        - 2 * lam * s["xy"]
        - 2 * alpha * s["y"]
        + 2 * alpha * lam * s["x"]
        + lam**2 * s["xx"]
        + alpha**2 * n
    )
    return 1.0 - ss_res / ss_tot


def mse_from_sums(s: pd.DataFrame, lam: pd.Series, alpha: pd.Series) -> pd.Series:
    n = s["count"]
    return (
        s["yy"]
        - 2 * lam * s["xy"]
        - 2 * alpha * s["y"]
        + 2 * alpha * lam * s["x"]
        + lam**2 * s["xx"]
        + alpha**2 * n
    ) / n


# ---------------------------------------------------------------------------
# Rolling baseline.
# ---------------------------------------------------------------------------
def rolling_baseline(
    sufficient_stats: pd.DataFrame,
    n_windows: int = 10,
    offset: int = 2,
) -> pd.DataFrame:
    """For each training month m = 1..n_windows, fit per-stock OLS and score
    on month m + offset (default 2-month gap, as in Anran's Section 2.2).

    Returns: DataFrame with columns
        [train_month, val_month, stock, lambda, alpha, is_r2, oos_r2].
    """
    records: list[dict] = []
    s = sufficient_stats
    for train_month in range(1, n_windows + 1):
        val_month = train_month + offset
        train = s.loc[s["month"] == train_month].groupby("stock")[STAT_COLS].sum()
        val = s.loc[s["month"] == val_month].groupby("stock")[STAT_COLS].sum()
        common = train.index.intersection(val.index)
        if len(common) == 0:
            continue
        train, val = train.loc[common], val.loc[common]

        lam, alpha = ols_from_sums(train)
        is_r2 = r2_from_sums(train, lam, alpha)
        oos_r2 = r2_from_sums(val, lam, alpha)

        for stock in common:
            records.append(
                {
                    "train_month": train_month,
                    "val_month": val_month,
                    "stock": stock,
                    "lambda": float(lam[stock]),
                    "alpha": float(alpha[stock]),
                    "is_r2": float(is_r2[stock]),
                    "oos_r2": float(oos_r2[stock]),
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Half-life grid search.
# ---------------------------------------------------------------------------
def half_life_grid_search(
    data: pd.DataFrame,
    daily_stats: pd.DataFrame,
    H_grid_minutes: Iterable[float],
    tau_bins: int,
    carry: CarryMode = "daily",
    model_types: Iterable[ModelType] = ("linear", "sqrt"),
    n_windows: int = 10,
    offset: int = 2,
    overnight_minutes: float = 16 * 60,
    start_time: str = "10:00:00",
    price_col: str = "mid",
    progress: bool = False,
) -> pd.DataFrame:
    """Recompute impact states for each H in the grid; fit rolling baseline;
    return DataFrame [model, H, train_month, stock, lambda, oos_r2].

    Note: this is the most expensive step (O(|H_grid| * stocks * months)),
    typically run once at the start of an analysis.
    """
    records: list[dict] = []
    for mt in model_types:
        for H in H_grid_minutes:
            if progress:
                print(f"  Grid: {mt}, H={H} min ...", end="\r", flush=True)
            impact_df = compute_impact_states(
                data, daily_stats, H, model_type=mt, overnight_minutes=overnight_minutes
            )
            feats = build_regression_features(
                impact_df,
                data,
                tau_bins,
                carry=carry,
                start_time=start_time,
                price_col=price_col,
            )
            stats = daily_sufficient_stats(feats)
            baseline = rolling_baseline(stats, n_windows=n_windows, offset=offset)
            baseline["model"] = mt
            baseline["H"] = H
            records.append(baseline)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Non-parametric binned estimator.
# ---------------------------------------------------------------------------
def build_bin_stats(
    features: pd.DataFrame,
    month: int,
    n_bins: int = 15,
    bin_edges: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-(stock, bin) sums of y and y^2 with shared quantile bins."""
    mdf = features.loc[features["month"] == month].copy()
    if bin_edges is None:
        bin_edges = np.quantile(mdf["x"].dropna(), np.linspace(0, 1, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    mdf["bin"] = pd.cut(mdf["x"], bins=bin_edges, labels=False, include_lowest=True)
    mdf = mdf.dropna(subset=["bin"])
    mdf["bin"] = mdf["bin"].astype(int)
    stats = (
        mdf.groupby(["stock", "bin"])
        .agg(sy=("y", "sum"), syy=("y", lambda v: (v**2).sum()), n=("y", "count"))
        .reset_index()
    )
    stats["mean_y"] = stats["sy"] / stats["n"]
    return stats, bin_edges


def universal_bin_means(train_stats: pd.DataFrame) -> pd.Series:
    pooled = train_stats.groupby("bin")[["sy", "n"]].sum()
    return (pooled["sy"] / pooled["n"]).rename("g_bar")


def regularised_bin_means(
    train_stats: pd.DataFrame, g_bar: pd.Series, gamma: float
) -> pd.DataFrame:
    merged = train_stats.merge(g_bar.rename("g_bar"), on="bin", how="left")
    merged["g_reg"] = (merged["sy"] + gamma * merged["g_bar"]) / (merged["n"] + gamma)
    return merged


def predict_and_score(
    test_stats: pd.DataFrame, g_reg_lookup: pd.DataFrame
) -> tuple[float, float]:
    merged = test_stats.merge(
        g_reg_lookup[["stock", "bin", "g_reg"]], on=["stock", "bin"], how="inner"
    )
    ss_res = (
        merged["syy"]
        - 2 * merged["g_reg"] * merged["sy"]
        + merged["g_reg"] ** 2 * merged["n"]
    )
    total_n = merged["n"].sum()
    total_sy = merged["sy"].sum()
    total_syy = merged["syy"].sum()
    if total_n == 0:
        return float("nan"), float("nan")
    y_bar = total_sy / total_n
    ss_tot = total_syy - total_n * y_bar**2
    total_res = float(ss_res.sum())
    r2 = 1 - total_res / ss_tot if ss_tot > 0 else float("nan")
    mse = total_res / total_n
    return mse, r2


@dataclass
class NonparametricFit:
    """Output of one rolling non-parametric estimation window."""

    train_month: int
    test_month: int
    val_month: int
    best_gamma: float
    median_n: float
    oos_r2_raw: float
    oos_r2_univ: float
    oos_r2_reg: float
    gamma_grid: np.ndarray
    gamma_mses: np.ndarray
    train_stats: pd.DataFrame
    g_bar: pd.Series
    reg_best: pd.DataFrame
    bin_edges: np.ndarray


def rolling_nonparametric(
    features: pd.DataFrame,
    n_bins: int = 15,
    n_windows: int = 10,
    gamma_log_range: tuple[float, float] = (-3.0, 3.0),
    gamma_n: int = 60,
) -> tuple[pd.DataFrame, dict[int, NonparametricFit]]:
    """Rolling non-parametric binned estimator with cross-stock regularisation.

    Windows: train month m, tune γ on m + 1, validate on m + 2 (Anran's choice).

    Returns
    -------
    summary : DataFrame with one row per (train_month) window.
    fits    : dict mapping train_month -> :class:`NonparametricFit`.
    """
    records: list[dict] = []
    fits: dict[int, NonparametricFit] = {}

    for train_month in range(1, n_windows + 1):
        test_month = train_month + 1
        val_month = train_month + 2
        train_stats, bin_edges = build_bin_stats(features, train_month, n_bins=n_bins)
        test_stats, _ = build_bin_stats(
            features, test_month, n_bins=n_bins, bin_edges=bin_edges
        )
        val_stats, _ = build_bin_stats(
            features, val_month, n_bins=n_bins, bin_edges=bin_edges
        )
        if train_stats.empty or test_stats.empty or val_stats.empty:
            continue

        g_bar = universal_bin_means(train_stats)
        median_n = float(train_stats["n"].median())
        gamma_grid = median_n * np.logspace(*gamma_log_range, gamma_n)

        gamma_mses = np.empty(len(gamma_grid))
        best_gamma, best_mse = gamma_grid[0], float("inf")
        for k, gamma in enumerate(gamma_grid):
            reg_df = regularised_bin_means(train_stats, g_bar, float(gamma))
            mse, _ = predict_and_score(test_stats, reg_df)
            gamma_mses[k] = mse
            if mse < best_mse:
                best_mse = mse
                best_gamma = float(gamma)

        reg_best = regularised_bin_means(train_stats, g_bar, best_gamma)
        raw = regularised_bin_means(train_stats, g_bar, 0.0)
        univ = regularised_bin_means(train_stats, g_bar, 1e15)

        _, oos_r2_reg = predict_and_score(val_stats, reg_best)
        _, oos_r2_raw = predict_and_score(val_stats, raw)
        _, oos_r2_univ = predict_and_score(val_stats, univ)

        records.append(
            {
                "train_month": train_month,
                "val_month": val_month,
                "best_gamma": best_gamma,
                "median_n": median_n,
                "oos_r2_raw": oos_r2_raw,
                "oos_r2_univ": oos_r2_univ,
                "oos_r2_reg": oos_r2_reg,
            }
        )
        fits[train_month] = NonparametricFit(
            train_month=train_month,
            test_month=test_month,
            val_month=val_month,
            best_gamma=best_gamma,
            median_n=median_n,
            oos_r2_raw=oos_r2_raw,
            oos_r2_univ=oos_r2_univ,
            oos_r2_reg=oos_r2_reg,
            gamma_grid=gamma_grid,
            gamma_mses=gamma_mses,
            train_stats=train_stats,
            g_bar=g_bar,
            reg_best=reg_best,
            bin_edges=bin_edges,
        )

    return pd.DataFrame(records), fits


def per_stock_lambda(baseline_df: pd.DataFrame) -> pd.Series:
    """Mean λ over all rolling windows, per stock. Use as `lam_lookup` for
    a scalar-λ backtest."""
    return baseline_df.groupby("stock")["lambda"].mean()


def per_stock_lambda_stats(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, and t-stat of λ over all rolling windows, per stock."""
    df = baseline_df.groupby("stock")["lambda"].agg(["mean", "std"])
    df["t_stat"] = df["mean"] / df["std"]
    return df

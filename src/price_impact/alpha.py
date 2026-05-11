"""Synthetic alpha generation (per project.ipynb cell 23).

Builds an unbiased, target-correlated alpha:

    α_t = x · r^h_t  +  y · (W_{t+h} - W_t) / P_t

with  x = 1  and  y = sqrt( (1/ρ² - 1) · Var(r^h) / E[P^{-2}] · 1/h )
chosen so that:

  1. E[r^h | α] = α               (unbiasedness — α is a fair forecast)
  2. Corr(α^h, r^h) = ρ            (target correlation)

Per project.ipynb, this is the "correct" version (x = 1), distinct from
complete_pipeline.ipynb's x = ρ² variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def create_synthetic_alpha(
    data: pd.DataFrame,
    rho: float = 0.05,
    h_bins: int = 1,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Per-stock synthetic alpha with target correlation `rho` and h-bin horizon.

    Parameters
    ----------
    data : DataFrame with columns [stock, date, time, mid].
    rho  : target correlation between α and forward h-bin return.
    h_bins : forecast horizon in 10-second bins.
    seed : RNG seed for the Wiener increments.

    Returns
    -------
    DataFrame with columns [stock, date, time, alpha, fwd_ret].
    """
    if not (0 < rho < 1):
        raise ValueError(f"rho must be in (0, 1); got {rho}")
    if h_bins < 1:
        raise ValueError("h_bins must be a positive integer")

    df = data[["stock", "date", "time", "mid"]].copy()
    df["fwd_ret"] = df.groupby(["stock", "date"])["mid"].transform(
        lambda p: p.pct_change(periods=h_bins).shift(-h_bins)
    )

    stock_params = (
        df.dropna(subset=["fwd_ret"])
        .groupby("stock")
        .agg(
            var_r=("fwd_ret", "var"),
            E_Pinv2=("mid", lambda p: (1.0 / p**2).mean()),
        )
    )
    stock_params["x"] = 1.0
    stock_params["y"] = np.sqrt(
        (1.0 / rho**2 - 1.0)
        * stock_params["var_r"]
        / stock_params["E_Pinv2"]
        / h_bins
    )

    df = df.merge(stock_params[["x", "y"]].reset_index(), on="stock", how="left")
    rng = np.random.default_rng(seed)
    df["dW"] = rng.standard_normal(len(df)) * np.sqrt(h_bins)
    df["alpha"] = df["x"] * df["fwd_ret"] + df["y"] * df["dW"] / df["mid"]
    df = df.dropna(subset=["alpha", "fwd_ret"])

    if verbose:
        emp = df.groupby("stock").apply(
            lambda g: pd.Series(
                {"var_alpha": g["alpha"].var(), "corr": g["alpha"].corr(g["fwd_ret"])}
            )
        )
        print(emp.round(6).to_string())
        print(
            f"Target ρ = {rho},  Mean empirical ρ = {emp['corr'].mean():.4f}"
        )

    return df[["stock", "date", "time", "alpha", "fwd_ret"]].reset_index(drop=True)

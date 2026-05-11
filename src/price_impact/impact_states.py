"""Normalised impact states Ī.

Two carry modes are produced **side by side** on the same bin-level panel,
so that downstream consumers (fitting, backtesting) can pick whichever they
need without recomputing:

    - `I_bar_daily` : reset Ī to 0 at the start of each (stock, date).
    - `I_bar_multi` : carry Ī across days within a stock, with an overnight
                      decay applied between sessions (no flow during the gap).

The recursion is the OW/AFS-style OU discretisation

        Ī_{t+1} = (1 - β) Ī_t + q̃_t,
        β = ln 2 / H_bins,   H_bins = half_life_minutes * 6  (10-s bins)

with model-specific normalised flow

        linear (OW):   q̃_t = σ_d * q_t / ADV_d
        sqrt  (AFS):   q̃_t = σ_d * sign(q_t) * sqrt(|q_t| / ADV_d).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ModelType = Literal["linear", "sqrt"]

BINS_PER_MINUTE = 6


def beta_from_half_life(half_life_minutes: float) -> float:
    return float(np.log(2.0) / (half_life_minutes * BINS_PER_MINUTE))


def decay_from_half_life(half_life_minutes: float) -> float:
    return 1.0 - beta_from_half_life(half_life_minutes)


def overnight_decay(half_life_minutes: float, overnight_minutes: float = 16 * 60) -> float:
    """Multiplicative decay applied to Ī across an overnight gap (no flow)."""
    if overnight_minutes <= 0:
        return 1.0
    return float(decay_from_half_life(half_life_minutes) ** (overnight_minutes * BINS_PER_MINUTE))


def q_tilde(orderflow: np.ndarray, sigma: float, adv: float, model_type: ModelType) -> np.ndarray:
    q = np.asarray(orderflow, dtype=float)
    if model_type == "linear":
        return sigma * q / adv
    return sigma * np.sign(q) * np.sqrt(np.abs(q) / adv)


def _ou_filter_daily(q_tilde_arr: np.ndarray, decay: float) -> np.ndarray:
    """Single-day OU recursion starting from Ī_0 = 0 (scipy lfilter)."""
    return lfilter([0.0, 1.0], [1.0, -decay], np.asarray(q_tilde_arr, dtype=float))


def _ou_filter_carry(q_tilde_arr: np.ndarray, decay: float, i0: float) -> np.ndarray:
    """OU recursion starting from a non-zero initial state."""
    qt = np.asarray(q_tilde_arr, dtype=float)
    n = qt.shape[0]
    out = np.empty(n)
    state = float(i0)
    for t in range(n):
        state = decay * state + qt[t]
        out[t] = state
    return out


def compute_impact_states(
    data: pd.DataFrame,
    daily_stats: pd.DataFrame,
    half_life_minutes: float,
    model_type: ModelType = "linear",
    overnight_minutes: float = 16 * 60,
    stock_col: str = "stock",
    date_col: str = "date",
    time_col: str = "time",
    order_flow_col: str = "orderFlow",
) -> pd.DataFrame:
    """Compute both daily-reset and multi-day impact states on the bin panel.

    Returns a DataFrame aligned to ``data`` with columns:
        [stock, date, time, q_tilde, I_bar_daily, I_bar_multi]
    """
    df = data[[stock_col, date_col, time_col, order_flow_col]].merge(
        daily_stats[["sigma", "ADV"]].reset_index(),
        on=[stock_col, date_col],
        how="inner",
    )
    df = df.sort_values([stock_col, date_col, time_col]).reset_index(drop=True)

    decay = decay_from_half_life(half_life_minutes)
    ovn = overnight_decay(half_life_minutes, overnight_minutes)

    # Vectorised q_tilde.
    if model_type == "linear":
        df["q_tilde"] = df["sigma"] * df[order_flow_col] / df["ADV"]
    else:
        df["q_tilde"] = (
            df["sigma"]
            * np.sign(df[order_flow_col])
            * np.sqrt(np.abs(df[order_flow_col]) / df["ADV"])
        )

    # Daily-reset Ī: lfilter per (stock, date) is vectorisable via groupby.
    df["I_bar_daily"] = df.groupby([stock_col, date_col])["q_tilde"].transform(
        lambda x: _ou_filter_daily(x.values, decay)
    )

    # Multi-day Ī: carry across days within a stock, with overnight decay.
    multi_parts: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for _, gs in df.groupby(stock_col, sort=False):
        i0 = 0.0
        for _, gd in gs.groupby(date_col, sort=False):
            gd_sorted = gd.sort_values(time_col)
            qt = gd_sorted["q_tilde"].to_numpy(dtype=float)
            i_day = _ou_filter_carry(qt, decay, i0)
            multi_parts.append(i_day)
            indices.append(gd_sorted.index.to_numpy())
            i0 = float(i_day[-1]) * ovn if len(i_day) else i0 * ovn

    flat_idx = np.concatenate(indices) if indices else np.empty(0, dtype=int)
    flat_vals = np.concatenate(multi_parts) if multi_parts else np.empty(0)
    multi = pd.Series(flat_vals, index=flat_idx, name="I_bar_multi")
    df["I_bar_multi"] = multi.reindex(df.index)

    cols = [stock_col, date_col, time_col, "q_tilde", "I_bar_daily", "I_bar_multi"]
    return df[cols]


def select_i_bar_column(carry: Literal["daily", "multi"]) -> str:
    if carry == "daily":
        return "I_bar_daily"
    if carry == "multi":
        return "I_bar_multi"
    raise ValueError(f"carry must be 'daily' or 'multi', got {carry!r}")

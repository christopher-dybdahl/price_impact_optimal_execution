"""Normalised impact states Ī.

Two carry modes are produced **side by side** on the same bin-level panel,
so that downstream consumers (fitting, backtesting) can pick whichever they
need without recomputing:

    - `I_bar_daily` : reset Ī to 0 at the start of each (stock, date).
    - `I_bar_multi` : carry Ī across days within a stock exactly as it ended
                      on the previous session, with no overnight decay.

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

ModelType = Literal["linear", "sqrt"]

BINS_PER_MINUTE = 6


def beta_from_half_life(half_life_minutes: float) -> float:
    return float(np.log(2.0) / (half_life_minutes * BINS_PER_MINUTE))


def decay_from_half_life(half_life_minutes: float) -> float:
    return 1.0 - beta_from_half_life(half_life_minutes)


def overnight_decay(half_life_minutes: float, overnight_minutes: float = 0.0) -> float:
    """Multiplicative decay applied to Ī across an overnight gap (no flow)."""
    if overnight_minutes <= 0:
        return 1.0
    return float(
        decay_from_half_life(half_life_minutes) ** (overnight_minutes * BINS_PER_MINUTE)
    )


def q_tilde(
    orderflow: np.ndarray, sigma: float, adv: float, c: float = 0.5
) -> np.ndarray:
    """Normalised flow: σ · sign(q) · |q/ADV|^c.

    Single point of truth for the AFS/OW impact normalisation.
    c=0.5 → AFS sqrt model.  c=1.0 → OW linear model (sign(q)·|q|^1 = q).
    """
    q = np.asarray(orderflow, dtype=float)
    return sigma * np.sign(q) * np.abs(q / adv) ** c


def _ou_filter_daily(q_tilde_arr: np.ndarray, decay: float) -> np.ndarray:
    """Single-day OU with Ī before the first bin = 0 (same recursion as multi on a reset day).

    Implemented via ``_ou_filter_carry(..., i0=0)`` so ``I_bar_daily`` matches
    ``I_bar_multi`` bit-for-bit on the first session day. SciPy ``lfilter`` is
    mathematically equivalent but can drift at ~1e-5 over ~2k bins from different
    float associativity.
    """
    return _ou_filter_carry(np.asarray(q_tilde_arr, dtype=float), decay, 0.0)


def _ou_filter_carry(q_tilde_arr: np.ndarray, decay: float, i0: float) -> np.ndarray:
    """OU recursion starting from a non-zero initial state.

    The first bin starts exactly from ``i0`` rather than applying a synthetic
    overnight decay step before the first observed flow.
    """
    qt = np.asarray(q_tilde_arr, dtype=float)
    n = qt.shape[0]
    out = np.empty(n)
    state = float(i0)
    for t in range(n):
        state = (state if t == 0 else decay * state) + qt[t]
        out[t] = state
    return out


def compute_impact_states(
    data: pd.DataFrame,
    daily_stats: pd.DataFrame,
    half_life_minutes: float,
    model_type: ModelType = "linear",
    c: float | None = None,
    overnight_minutes: float = 0.0,
    stock_col: str = "stock",
    date_col: str = "date",
    time_col: str = "time",
    order_flow_col: str = "trade",
) -> pd.DataFrame:
    """Compute both daily-reset and multi-day impact states on the bin panel.

    ``overnight_minutes`` is retained for API compatibility. Multi-day carry
    intentionally applies no overnight decay: the next session starts from the
    previous session's closing impact state, as if trading were continuous.

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
    _ = overnight_minutes

    # Resolve c from explicit arg, falling back to model_type convention.
    if c is None:
        c = 1.0 if model_type == "linear" else 0.5

    # Vectorised q_tilde — single formula, consistent with the q_tilde() function.
    df["q_tilde"] = (
        df["sigma"]
        * np.sign(df[order_flow_col])
        * np.abs(df[order_flow_col] / df["ADV"]) ** c
    )

    # Daily-reset Ī: same inner loop as multi (i0=0 each day) for numerical agreement.
    df["I_bar_daily"] = df.groupby([stock_col, date_col])["q_tilde"].transform(
        lambda x: _ou_filter_daily(x.values, decay)
    )

    # Multi-day Ī: carry across days within a stock without overnight decay.
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
            i0 = float(i_day[-1]) if len(i_day) else i0

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

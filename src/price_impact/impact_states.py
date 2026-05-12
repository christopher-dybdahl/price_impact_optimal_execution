"""Normalised impact states Ī.

Two carry modes are produced **side by side** on the same bin-level panel,
so that downstream consumers (fitting, backtesting) can pick whichever they
need without recomputing:

    - `I_bar_daily` : reset Ī to 0 at the start of each (stock, date).
    - `I_bar_multi` : carry Ī across days within a stock exactly as it ended
                      on the previous session, with no overnight decay.

Two flavours of the recursion live here:

1. **Legacy** (``compute_impact_states``) — OU on the σ-scaled normalised flow,
   with the model-specific transform applied *inside* the recursion. The sqrt
   branch in this form is **not** canonical Alfonsi–Fruth–Schied — see
   ``compute_impact_states_concave`` for the canonical version.

       Ī_{t+1} = (1 - β) Ī_t + q̃_t
       linear (OW):   q̃_t = σ_d * q_t / ADV_d
       sqrt  (legacy): q̃_t = σ_d * sign(q_t) * sqrt(|q_t| / ADV_d)

2. **Canonical AFS-family** (``compute_impact_states_concave``) — linear OU
   on q/ADV with the concave transform applied *outside*:

       J_{t+1} = (1 - β) J_t + q_t / ADV_d
       Ī_t    = σ_d · sign(J_t) · |J_t|^c

   c = 1   → linear (OW-equivalent on a daily-reset day, since σ is constant
            within the day and factors out of the linear OU).
   c = 0.5 → canonical AFS (Alfonsi–Fruth–Schied square-root).
   c ∈ (0.5, 1) → generalised concave family between them.
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
    orderflow: np.ndarray, sigma: float, adv: float, model_type: ModelType
) -> np.ndarray:
    q = np.asarray(orderflow, dtype=float)
    if model_type == "linear":
        return sigma * q / adv
    return sigma * np.sign(q) * np.sqrt(np.abs(q) / adv)


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

    # Vectorised q_tilde.
    if model_type == "linear":
        df["q_tilde"] = df["sigma"] * df[order_flow_col] / df["ADV"]
    else:
        df["q_tilde"] = (
            df["sigma"]
            * np.sign(df[order_flow_col])
            * np.sqrt(np.abs(df[order_flow_col]) / df["ADV"])
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


# ---------------------------------------------------------------------------
# Canonical AFS-family: linear OU on q/ADV, concavity α on the outside.
# ---------------------------------------------------------------------------
def _lfilter_ou_daily(series: pd.Series, decay: float) -> np.ndarray:
    """Vectorised per-group OU filter via scipy.signal.lfilter.

    Equivalent to ``state ← decay·state + x[t]`` with state reset to 0 at the
    start of each group. ~100× faster than the Python loop and matches it to
    O(1e-5) — fine for grid search where we compare R² values, not paths.
    """
    from scipy.signal import lfilter

    return lfilter([1.0], [1.0, -decay], series.to_numpy(dtype=float))


def _lfilter_ou_carry(series: pd.Series, decay: float) -> np.ndarray:
    """OU filter across the whole series, no reset between contiguous groups.

    Used for the multi-day carry: state at the start of day d is
    ``decay·state_end_of_(d-1) + 0`` (lfilter's natural behaviour), which
    differs from the legacy ``_ou_filter_carry`` only at day boundaries
    (legacy adds the first-bin flow without decaying the carry). The effect
    is one bin per day, so on R² comparisons it is negligible.
    """
    from scipy.signal import lfilter

    return lfilter([1.0], [1.0, -decay], series.to_numpy(dtype=float))


def compute_impact_states_concave(
    data: pd.DataFrame,
    daily_stats: pd.DataFrame,
    half_life_minutes: float,
    c: float,
    stock_col: str = "stock",
    date_col: str = "date",
    time_col: str = "time",
    order_flow_col: str = "trade",
) -> pd.DataFrame:
    """Canonical AFS-family impact states.

    Linear OU on q/ADV; concave transform applied to the aggregated state:

        J_t  = (1 - β)·J_{t-1} + q_t / ADV_d
        Ī_t = σ_d · sign(J_t) · |J_t|^c

    c = 1   → linear (OW-equivalent on daily-reset; σ factors out of OU).
    c = 0.5 → canonical Alfonsi–Fruth–Schied square-root impact.

    Returns
    -------
    DataFrame aligned to ``data`` with columns
        [stock, date, time, sigma, ADV, J_daily, J_multi,
         I_bar_daily, I_bar_multi]

    The Ī columns are the only ones consumed by ``build_regression_features``;
    J columns are exposed so the same H-cache can be reused across c values
    (concavity is a pointwise pow(), so the expensive OU only runs once per H).
    """
    df = data[[stock_col, date_col, time_col, order_flow_col]].merge(
        daily_stats[["sigma", "ADV"]].reset_index(),
        on=[stock_col, date_col],
        how="inner",
    )
    df = df.sort_values([stock_col, date_col, time_col]).reset_index(drop=True)

    decay = decay_from_half_life(half_life_minutes)

    # Dimensionless linear flow per bin (σ stays outside the OU).
    df["q_lin"] = df[order_flow_col] / df["ADV"]

    # Daily-reset linear OU on q_lin.
    df["J_daily"] = df.groupby([stock_col, date_col])["q_lin"].transform(
        lambda g: _lfilter_ou_daily(g, decay)
    )

    # Multi-day linear OU on q_lin (one continuous filter per stock).
    df["J_multi"] = df.groupby(stock_col, sort=False)["q_lin"].transform(
        lambda g: _lfilter_ou_carry(g, decay)
    )

    # Apply the concave transform on the aggregated state.
    for j_col, i_col in (("J_daily", "I_bar_daily"), ("J_multi", "I_bar_multi")):
        J = df[j_col].to_numpy(dtype=float)
        df[i_col] = df["sigma"].to_numpy(dtype=float) * np.sign(J) * np.power(np.abs(J), c)

    out_cols = [
        stock_col, date_col, time_col,
        "sigma", "ADV",
        "J_daily", "J_multi",
        "I_bar_daily", "I_bar_multi",
    ]
    return df[out_cols]

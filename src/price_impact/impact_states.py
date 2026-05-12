"""Canonical AFS-family impact states — single point of truth.

The impact model is the canonical Alfonsi--Fruth--Schied specification:

    J_{t+1} = (1 - β)·J_t + q_t / ADV_d        (linear OU on q / ADV)
    Ī_t    = σ_d · sign(J_t) · |J_t|^c          (power-law shape applied OUTSIDE)

where the two carry modes are produced side by side on the same bin-level panel:

    - ``I_bar_daily`` : reset J to 0 at the start of each (stock, date).
    - ``I_bar_multi`` : carry J across days within a stock with no overnight decay.

Convention:
    c = 1.0 → OW linear. Note: for c=1 the σ factors out of the OU, so
              Ī ≡ σ·J, which is identical to the legacy linear OW formula
              q̃ = σ·q/ADV inside the OU.
    c = 0.5 → canonical AFS square-root.

Two pure helpers below — ``apply_concavity`` and ``invert_concavity`` — are
imported by ``backtest.waelbroeck_prices`` and ``strategy.optimal_strategy``
so all call-sites share **one** formula.  Add a new c value here, every
downstream consumer follows.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ModelType = Literal["linear", "sqrt"]

BINS_PER_MINUTE = 6


# ---------------------------------------------------------------------------
# Half-life / decay helpers.
# ---------------------------------------------------------------------------
def beta_from_half_life(half_life_minutes: float) -> float:
    return float(np.log(2.0) / (half_life_minutes * BINS_PER_MINUTE))


def decay_from_half_life(half_life_minutes: float) -> float:
    return 1.0 - beta_from_half_life(half_life_minutes)


def overnight_decay(half_life_minutes: float, overnight_minutes: float = 0.0) -> float:
    """Multiplicative decay over an overnight gap (kept for API compatibility)."""
    if overnight_minutes <= 0:
        return 1.0
    return float(
        decay_from_half_life(half_life_minutes) ** (overnight_minutes * BINS_PER_MINUTE)
    )


# ---------------------------------------------------------------------------
# Single point of truth: canonical concavity layer.
# ---------------------------------------------------------------------------
def apply_concavity(j, sigma, c: float):
    """Ī = σ · sign(J) · |J|^c  (canonical AFS shape, applied OUTSIDE the OU).

    Use after computing J = OU(q/ADV). c=1 → linear (Ī = σ·J); c=0.5 → sqrt.
    Vectorised; ``j`` can be scalar or array, ``sigma`` scalar or array of matching shape.
    """
    j = np.asarray(j, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    return sigma_arr * np.sign(j) * np.power(np.abs(j), c)


def invert_concavity(ibar, sigma, c: float):
    """J = sign(Ī) · (|Ī| / σ)^(1/c)  — exact inverse of ``apply_concavity``.

    Used by ``strategy.optimal_strategy`` to convert a target Ī* into the J*
    state to drive, and by ``backtest.waelbroeck_prices`` to convert a carried
    Ī initial state into the J initial state for the next day's OU.
    """
    ibar = np.asarray(ibar, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if np.any(sigma_arr <= 0):
        return np.zeros_like(ibar) if ibar.ndim else np.float64(0.0)
    return np.sign(ibar) * np.power(np.abs(ibar) / sigma_arr, 1.0 / c)


def resolve_c(model_type: ModelType | None, c: float | None) -> float:
    """Resolve concavity from (model_type, c) inputs. c wins when set."""
    if c is not None:
        return float(c)
    if model_type == "linear":
        return 1.0
    if model_type == "sqrt":
        return 0.5
    raise ValueError(f"specify c or model_type ∈ {{'linear','sqrt'}}, got {model_type!r}")


# ---------------------------------------------------------------------------
# Legacy alias kept for back-compat imports. The ARGUMENT semantics are now
# canonical: this is the σ-scaled flow ratio raised to power c, useful only
# for per-bin display. The canonical Ī uses ``apply_concavity`` instead.
# ---------------------------------------------------------------------------
def q_tilde(orderflow, sigma, adv, c: float = 1.0):
    """Per-bin σ-scaled flow ratio σ·sign(q)·|q/ADV|^c.

    Kept for backward-compat with old call-sites. In the canonical AFS
    pipeline this function is **not** the input to the OU — the OU runs on
    the dimensionless ``q/ADV`` and ``apply_concavity`` is then applied
    OUTSIDE the OU.
    """
    q = np.asarray(orderflow, dtype=float)
    return sigma * np.sign(q) * np.power(np.abs(q / adv), c)


# ---------------------------------------------------------------------------
# OU filter helpers.
# ---------------------------------------------------------------------------
def _ou_filter_daily(x: np.ndarray, decay: float) -> np.ndarray:
    """Single-day OU starting from state 0 (Python loop — bit-exact)."""
    return _ou_filter_carry(np.asarray(x, dtype=float), decay, 0.0)


def _ou_filter_carry(x: np.ndarray, decay: float, i0: float) -> np.ndarray:
    """OU recursion starting from non-zero initial state ``i0``."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    out = np.empty(n)
    state = float(i0)
    for t in range(n):
        state = (state if t == 0 else decay * state) + x[t]
        out[t] = state
    return out


def _lfilter_ou(series: pd.Series, decay: float) -> np.ndarray:
    """Vectorised OU via scipy.signal.lfilter (state resets to 0 each call)."""
    from scipy.signal import lfilter

    return lfilter([1.0], [1.0, -decay], series.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Main panel-builder: canonical impact states.
# ---------------------------------------------------------------------------
def compute_impact_states(
    data: pd.DataFrame,
    daily_stats: pd.DataFrame,
    half_life_minutes: float,
    model_type: ModelType | None = "linear",
    c: float | None = None,
    overnight_minutes: float = 0.0,
    stock_col: str = "stock",
    date_col: str = "date",
    time_col: str = "time",
    order_flow_col: str = "trade",
) -> pd.DataFrame:
    """Canonical AFS-family impact states (single source of truth).

    Linear OU on q/ADV; concave transform applied to the aggregated state:

        J_t = (1 - β)·J_{t-1} + q_t / ADV_d
        Ī_t = σ_d · sign(J_t) · |J_t|^c                ← ``apply_concavity``

    c is taken from the ``c`` argument if provided, else derived from
    ``model_type`` (linear → 1.0, sqrt → 0.5). ``overnight_minutes`` is
    accepted for API compatibility; multi-day carry does not apply overnight
    decay.

    Returns
    -------
    DataFrame aligned to ``data`` with columns
        [stock, date, time, sigma, ADV, J_daily, J_multi, I_bar_daily, I_bar_multi]
    """
    c_val = resolve_c(model_type, c)
    _ = overnight_minutes

    df = data[[stock_col, date_col, time_col, order_flow_col]].merge(
        daily_stats[["sigma", "ADV"]].reset_index(),
        on=[stock_col, date_col],
        how="inner",
    )
    df = df.sort_values([stock_col, date_col, time_col]).reset_index(drop=True)

    decay = decay_from_half_life(half_life_minutes)

    # Dimensionless linear flow per bin. σ stays OUTSIDE the OU.
    df["q_lin"] = df[order_flow_col] / df["ADV"]

    # Daily-reset linear OU on q_lin.
    df["J_daily"] = df.groupby([stock_col, date_col])["q_lin"].transform(
        lambda g: _lfilter_ou(g, decay)
    )

    # Multi-day linear OU on q_lin (one continuous filter per stock).
    df["J_multi"] = df.groupby(stock_col, sort=False)["q_lin"].transform(
        lambda g: _lfilter_ou(g, decay)
    )

    # Apply the concave transform on the aggregated state.
    sig = df["sigma"].to_numpy(dtype=float)
    df["I_bar_daily"] = apply_concavity(df["J_daily"].to_numpy(dtype=float), sig, c_val)
    df["I_bar_multi"] = apply_concavity(df["J_multi"].to_numpy(dtype=float), sig, c_val)

    return df[[
        stock_col, date_col, time_col,
        "sigma", "ADV",
        "J_daily", "J_multi",
        "I_bar_daily", "I_bar_multi",
    ]]


def select_i_bar_column(carry: Literal["daily", "multi"]) -> str:
    if carry == "daily":
        return "I_bar_daily"
    if carry == "multi":
        return "I_bar_multi"
    raise ValueError(f"carry must be 'daily' or 'multi', got {carry!r}")


# Backward-compat alias used by fitting.concavity_grid_search and elsewhere.
def compute_impact_states_concave(
    data: pd.DataFrame,
    daily_stats: pd.DataFrame,
    half_life_minutes: float,
    c: float,
    **kwargs,
) -> pd.DataFrame:
    """Alias for :func:`compute_impact_states` with explicit ``c``."""
    return compute_impact_states(
        data, daily_stats, half_life_minutes, model_type=None, c=c, **kwargs
    )

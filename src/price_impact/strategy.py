"""Optimal trading strategies, model-aware.

We expose three strategies:

  * :func:`ow_optimal_strategy`   — OW (linear) closed form.
  * :func:`afs_optimal_strategy`  — AFS (sqrt) closed form. Inverts the
                                    sqrt impact recursion to recover the
                                    optimal trade per bin given a target
                                    normalized impact state.
  * :func:`ext_ow_optimal_strategy_timedep_lambda` — placeholder for the
                                    extended OW model with time-varying λ(t).
                                    Not implemented; raises NotImplementedError.

The "model" object pattern lets a downstream backtest pick the right
strategy generically without branching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

ModelType = Literal["linear", "sqrt"]

BINS_PER_MINUTE = 6


# ---------------------------------------------------------------------------
# OW (linear λ) closed-form strategy.
# ---------------------------------------------------------------------------
def ow_optimal_strategy(
    alpha: np.ndarray,
    sigma: float,
    adv: float,
    lam: float,
    half_life_minutes: float,
    max_position_adv: float = 0.005,
    liquidation_minutes: int = 30,
    ibar_init: float = 0.0,
) -> np.ndarray:
    """OW-optimal intraday execution.

    Target normalized impact state:
        Ī*_t = α_t / (2λ)   so that  λ·Ī*_t = α_t/2  (HJB: pay half alpha in impact)

    Trade per bin recovered by inverting the linear impact recursion:
        decay = 1 − β,   β = ln 2 / H_bins
        z_t   = Ī*_t − decay · Ī_{t−1}
        q*_t  = ADV · z_t / σ              (linear inversion of q̃ = σ q / ADV)

    Structurally identical to AFS — target an impact state, invert the recursion —
    but uses the linear inversion instead of the sqrt one.  The loop tracks the
    actual impact state Ī_prev from the clipped trade so the recursion inversion
    remains exact when the position cap binds.
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    n = alpha.shape[0]
    if lam <= 0 or n == 0 or sigma <= 0 or adv <= 0 or half_life_minutes <= 0:
        return np.zeros(n)

    H_bins = half_life_minutes * BINS_PER_MINUTE
    beta = np.log(2.0) / H_bins
    decay = 1.0 - beta

    ibar_star = alpha / (2.0 * lam)

    liq_bins = min(liquidation_minutes * BINS_PER_MINUTE, n)
    if liq_bins > 0:
        ramp = np.ones(n)
        ramp[n - liq_bins :] = np.linspace(1.0, 0.0, liq_bins)
        ibar_star = ibar_star * ramp

    max_pos = max_position_adv * adv
    pos = 0.0
    ibar_prev = float(ibar_init)
    trades = np.zeros(n)
    for t in range(n):
        z = ibar_star[t] - decay * ibar_prev
        q_desired = adv * z / sigma
        new_pos = float(np.clip(pos + q_desired, -max_pos, max_pos))
        q = new_pos - pos
        pos = new_pos
        trades[t] = q
        ibar_prev = decay * ibar_prev + sigma * q / adv
    return trades


# ---------------------------------------------------------------------------
# AFS (sqrt) strategy.
# ---------------------------------------------------------------------------
def afs_optimal_strategy(
    alpha: np.ndarray,
    sigma: float,
    adv: float,
    lam: float,
    half_life_minutes: float,
    max_position_adv: float = 0.005,
    liquidation_minutes: int = 30,
    c: float = 0.5,
    ibar_init: float = 0.0,
) -> np.ndarray:
    """AFS (sqrt) optimal execution.

    Target normalized impact state (concavity c = 0.5 for sqrt AFS):
        Ī*_t = α_t / ((1 + c) · λ)

    Trade per bin recovered by inverting the AFS impact recursion:
        decay = 1 − β,   β = ln 2 / H_bins
        z_t   = Ī*_t − decay · Ī_{t−1}
        q*_t  = ADV · sign(z_t) · (|z_t| / σ)²

    The end-of-day ramp is applied to Ī* before the loop.  AFS differs
    from OW only in the inversion step: sqrt impact requires the quadratic
    inversion  q = ADV · sign(z) · (|z|/σ)²,  whereas linear OW uses the
    linear inversion  q = ADV · z / σ.  Both strategies track cumulative
    position and clip to ±`max_position_adv`·ADV, then update Ī_prev from
    the actual clipped trade so the recursion inversion remains exact.
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    n = alpha.shape[0]
    if lam <= 0 or n == 0 or sigma <= 0 or adv <= 0 or c <= 0 or half_life_minutes <= 0:
        return np.zeros(n)

    H_bins = half_life_minutes * BINS_PER_MINUTE
    beta = np.log(2.0) / H_bins
    decay = 1.0 - beta

    ibar_star = alpha / ((1.0 + c) * lam)

    liq_bins = min(liquidation_minutes * BINS_PER_MINUTE, n)
    if liq_bins > 0:
        ramp = np.ones(n)
        ramp[n - liq_bins :] = np.linspace(1.0, 0.0, liq_bins)
        ibar_star = ibar_star * ramp

    max_pos = max_position_adv * adv
    pos = 0.0
    ibar_prev = float(ibar_init)
    trades = np.zeros(n)
    for t in range(n):
        z = ibar_star[t] - decay * ibar_prev
        q_desired = adv * np.sign(z) * (abs(z) / sigma) ** 2
        new_pos = float(np.clip(pos + q_desired, -max_pos, max_pos))
        q = new_pos - pos
        pos = new_pos
        trades[t] = q
        q_tilde = sigma * np.sign(q) * np.sqrt(abs(q) / adv)
        ibar_prev = decay * ibar_prev + q_tilde
    return trades


# ---------------------------------------------------------------------------
# Extended OW with time-dependent λ — placeholder.
# ---------------------------------------------------------------------------
def ext_ow_optimal_strategy_timedep_lambda(
    alpha: np.ndarray,
    sigma: float,
    adv: float,
    lam_t: np.ndarray,
    half_life_minutes: float,
    max_position_adv: float = 0.005,
    liquidation_minutes: int = 30,
) -> np.ndarray:
    """Placeholder for the extended OW model where λ varies through the day.

    The closed-form target position becomes  X*_t = α_t · ADV / (2 λ(t) σ),
    but the HJB-optimal trade rule is no longer κ-constant: it depends on
    the path of λ. Wire the proper derivation in here once it is settled.
    """
    raise NotImplementedError(
        "Extended OW with time-dependent λ is not implemented; this is a"
        " placeholder for the closed-form solution to be added later."
    )


# ---------------------------------------------------------------------------
# Strategy dispatch.
# ---------------------------------------------------------------------------
StrategyFn = Callable[..., np.ndarray]

STRATEGIES: dict[str, StrategyFn] = {
    "ow": ow_optimal_strategy,
    "linear": ow_optimal_strategy,
    "afs": afs_optimal_strategy,
    "sqrt": afs_optimal_strategy,
    "ext_ow": ext_ow_optimal_strategy_timedep_lambda,
}


def get_strategy(name: str) -> StrategyFn:
    key = name.lower()
    if key not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[key]


# ---------------------------------------------------------------------------
# Impact-model handle — what the backtest needs to know.
# ---------------------------------------------------------------------------
@dataclass
class ImpactModel:
    """Bundles the impact-model identity for a backtest run.

    `lam_lookup` is per-stock; if `lam_t_lookup` is set, the model is treated
    as time-dependent (one array per stock-day), and the simulator uses
    g_t = λ_t * Ī_t instead of g = λ * Ī.
    """

    model_type: ModelType  # 'linear' or 'sqrt'
    half_life_minutes: float
    lam_lookup: dict[str, float]  # stock -> scalar λ
    lam_t_lookup: dict[tuple[str, object], np.ndarray] | None = None
    strategy: str = "ow"  # 'ow' | 'afs' | 'ext_ow'

    def is_time_dependent(self) -> bool:
        return self.lam_t_lookup is not None

    def lam_for(self, stock: str) -> float:
        return float(self.lam_lookup.get(stock, float("nan")))

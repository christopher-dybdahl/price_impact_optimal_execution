"""Optimal trading strategies, model-aware.

Single point of truth: all strategies delegate to :func:`optimal_strategy`,
which parameterises both OW (c=1) and AFS (c=0.5) via the concavity exponent c.

The normalised flow  q̃ = σ · sign(q) · |q/ADV|^c  is the same formula used
in ``impact_states.q_tilde`` and in ``waelbroeck_prices`` — one definition,
three call-sites.

Exposed strategies:
  * :func:`optimal_strategy`  — unified, takes c as parameter.
  * :func:`ow_optimal_strategy`  — thin wrapper: c=1.
  * :func:`afs_optimal_strategy` — thin wrapper: c=0.5 (or user-supplied c).
  * :func:`ext_ow_optimal_strategy_timedep_lambda` — placeholder, raises NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

ModelType = Literal["linear", "sqrt"]

BINS_PER_MINUTE = 6


# ---------------------------------------------------------------------------
# Unified optimal strategy — single point of truth for all c values.
# ---------------------------------------------------------------------------
def optimal_strategy(
    alpha: np.ndarray,
    sigma: float,
    adv: float,
    lam: float,
    half_life_minutes: float,
    c: float = 0.5,
    max_position_adv: float = 0.005,
    liquidation_minutes: int = 30,
    ibar_init: float = 0.0,
) -> np.ndarray:
    """Optimal execution for a power-law impact model with concavity exponent c.

    Normalised flow (one point of truth, mirrors impact_states.q_tilde):
        q̃ = σ · sign(q) · |q/ADV|^c

    Inversion (algebraic inverse of q̃):
        q = ADV · sign(z) · |z/σ|^(1/c)

    HJB target impact state:
        Ī*_t = α_t / ((1 + c) · λ)
        c = 0.5 (AFS sqrt)  → target = 2/3 · α/λ
        c = 1.0 (OW linear) → target = 1/2 · α/λ

    ## TODO: For c < 1, the actual price impact produced by our trades in the
    ## Waelbroeck simulator is delta_g = lam*(OU(q_tilde(q_agg,c)) − OU(q_tilde(q_others,c))),
    ## which differs from the standalone self-tracking below because q_tilde is
    ## nonlinear (concave). Correcting this requires knowing q_others at trade
    ## time, which is not available to the strategy. The approximation is exact
    ## for c = 1 (OW, linear additivity) and approximate for c < 1 (AFS).
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    n = alpha.shape[0]
    if lam <= 0 or n == 0 or sigma <= 0 or adv <= 0 or c <= 0 or half_life_minutes <= 0:
        return np.zeros(n)

    H_bins = half_life_minutes * BINS_PER_MINUTE
    decay = 1.0 - np.log(2.0) / H_bins

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
        q_desired = adv * float(np.sign(z)) * (abs(z) / sigma) ** (1.0 / c)
        new_pos = float(np.clip(pos + q_desired, -max_pos, max_pos))
        q = new_pos - pos
        pos = new_pos
        trades[t] = q
        # Inline q_tilde formula — must match impact_states.q_tilde(q, sigma, adv, c).
        ibar_prev = decay * ibar_prev + float(sigma * np.sign(q) * abs(q / adv) ** c)
    return trades


# ---------------------------------------------------------------------------
# Named wrappers — thin delegates to optimal_strategy.
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
    **_,
) -> np.ndarray:
    """OW (linear, c=1) optimal strategy. Delegates to optimal_strategy."""
    return optimal_strategy(
        alpha, sigma, adv, lam, half_life_minutes,
        c=1.0, max_position_adv=max_position_adv,
        liquidation_minutes=liquidation_minutes, ibar_init=ibar_init,
    )


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
    **_,
) -> np.ndarray:
    """AFS (sqrt by default, c=0.5) optimal strategy. Delegates to optimal_strategy."""
    return optimal_strategy(
        alpha, sigma, adv, lam, half_life_minutes,
        c=c, max_position_adv=max_position_adv,
        liquidation_minutes=liquidation_minutes, ibar_init=ibar_init,
    )


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
    """Placeholder for the extended OW model where λ varies through the day."""
    _ = (alpha, sigma, adv, lam_t, half_life_minutes, max_position_adv, liquidation_minutes)
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

    `c` is the concavity exponent used by both the strategy (optimal_strategy)
    and the simulator (waelbroeck_prices / impact_states.q_tilde) — one value,
    one formula, two call-sites.  c=0.5 → AFS sqrt; c=1.0 → OW linear.

    `model_type` is kept for labelling / compatibility; it is NOT used in the
    q_tilde computation (c is the authoritative parameter).

    `lam_t_lookup` makes the model time-dependent (one array per stock-day).
    """

    model_type: ModelType  # 'linear' or 'sqrt' — label only; c is authoritative
    half_life_minutes: float
    lam_lookup: dict[str, float]  # stock -> scalar λ
    c: float = 0.5               # concavity exponent: 0.5 (AFS) or 1.0 (OW)
    lam_t_lookup: dict[tuple[str, object], np.ndarray] | None = None
    strategy: str = "ow"  # 'ow' | 'afs' | 'ext_ow'

    def is_time_dependent(self) -> bool:
        return self.lam_t_lookup is not None

    def lam_for(self, stock: str) -> float:
        return float(self.lam_lookup.get(stock, float("nan")))

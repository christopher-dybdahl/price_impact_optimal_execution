"""Optimal trading strategies, model-aware.

Single point of truth: all strategies delegate to :func:`optimal_strategy`,
which parameterises both OW (c=1) and AFS (c=0.5) via the concavity exponent c.

The impact model is the canonical Alfonsi–Fruth–Schied form:

    J_{t+1} = (1 - β)·J_t + q_t / ADV          (linear OU on q/ADV)
    Ī_t    = σ · sign(J_t) · |J_t|^c            (power-law shape OUTSIDE)

``apply_concavity`` and ``invert_concavity`` from :mod:`impact_states` are
the centralised helpers used here, in ``backtest.waelbroeck_prices``, and
in :func:`compute_impact_states` — one definition, three call-sites.

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

from .impact_states import apply_concavity, invert_concavity

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
    """Optimal execution for the canonical AFS-family impact model.

    Impact specification (matches :mod:`impact_states`):
        J_{t+1} = (1 - β)·J_t + q_t / ADV
        Ī_t    = σ · sign(J_t) · |J_t|^c        (via ``apply_concavity``)

    HJB target impact state (one-half-α rule generalised to concavity c):
        Ī*_t = α_t / ((1 + c) · λ)
        c = 1.0 (OW)   → target = α/(2λ)
        c = 0.5 (AFS)  → target = 2α/(3λ)

    Inversion — two layers:
        1. Ī* → J*  via :func:`impact_states.invert_concavity`
           J*_t = sign(Ī*_t) · (|Ī*_t| / σ)^(1/c)
        2. J* → q  by linear inversion of the OU step:
           q_t = ADV · (J*_t − decay · J_{t−1})

    Trades are clipped to ±``max_position_adv``·ADV; the actual (post-clip)
    J state is tracked each bin so the inversion stays exact at the cap.
    ``ibar_init`` is interpreted as a carried Ī from the previous session
    (OW and AFS callers use the same parameter name); on entry it is
    inverted to a J state via :func:`impact_states.invert_concavity`.
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    n = alpha.shape[0]
    if lam <= 0 or n == 0 or sigma <= 0 or adv <= 0 or c <= 0 or half_life_minutes <= 0:
        return np.zeros(n)

    H_bins = half_life_minutes * BINS_PER_MINUTE
    decay = 1.0 - np.log(2.0) / H_bins

    # Target Ī*, then ramp to zero over the EOD liquidation window.
    ibar_star = alpha / ((1.0 + c) * lam)
    liq_bins = min(liquidation_minutes * BINS_PER_MINUTE, n)
    if liq_bins > 0:
        ramp = np.ones(n)
        ramp[n - liq_bins :] = np.linspace(1.0, 0.0, liq_bins)
        ibar_star = ibar_star * ramp

    # Map target Ī* → target J* via the centralised inverse-concavity helper.
    jbar_star = np.asarray(invert_concavity(ibar_star, sigma, c), dtype=float)

    # Carry-in: ibar_init is in Ī units; invert to a J initial state.
    jbar_prev = (
        float(invert_concavity(ibar_init, sigma, c)) if ibar_init != 0.0 else 0.0
    )

    max_pos = max_position_adv * adv
    pos = 0.0
    trades = np.zeros(n)
    for t in range(n):
        # Linear OU inversion: target J change z drives a linear trade.
        z = jbar_star[t] - decay * jbar_prev
        q_desired = adv * z
        new_pos = float(np.clip(pos + q_desired, -max_pos, max_pos))
        q = new_pos - pos
        pos = new_pos
        trades[t] = q
        # J evolves with the actual clipped trade (linear in q for J).
        jbar_prev = decay * jbar_prev + q / adv
    # Ī is recovered downstream via apply_concavity(J, σ, c) when needed.
    _ = apply_concavity  # imported for the centralised symbol contract.
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
        alpha,
        sigma,
        adv,
        lam,
        half_life_minutes,
        c=1.0,
        max_position_adv=max_position_adv,
        liquidation_minutes=liquidation_minutes,
        ibar_init=ibar_init,
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
        alpha,
        sigma,
        adv,
        lam,
        half_life_minutes,
        c=c,
        max_position_adv=max_position_adv,
        liquidation_minutes=liquidation_minutes,
        ibar_init=ibar_init,
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
    _ = (
        alpha,
        sigma,
        adv,
        lam_t,
        half_life_minutes,
        max_position_adv,
        liquidation_minutes,
    )
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

    ``c`` is the concavity exponent used by the canonical pipeline:

        J_{t+1} = (1 - β)·J_t + q_t / ADV          (linear OU on q/ADV)
        Ī_t    = σ · sign(J_t) · |J_t|^c           (apply_concavity)

    The same ``c`` is consumed by :func:`optimal_strategy`, by
    :func:`backtest.waelbroeck_prices`, and by
    :func:`impact_states.compute_impact_states` — one value, one formula,
    three call-sites.  c=0.5 → AFS sqrt; c=1.0 → OW linear.

    ``model_type`` is kept for labelling / compatibility; ``c`` is the
    authoritative parameter and is what flows into the canonical helpers.

    ``lam_t_lookup`` makes the model time-dependent (one array per stock-day).
    """

    model_type: ModelType  # 'linear' or 'sqrt' — label only; c is authoritative
    half_life_minutes: float
    lam_lookup: dict[str, float]  # stock -> scalar λ
    c: float = 0.5  # concavity exponent: 0.5 (AFS) or 1.0 (OW)
    lam_t_lookup: dict[tuple[str, object], np.ndarray] | None = None
    strategy: str = "ow"  # 'ow' | 'afs' | 'ext_ow'

    def is_time_dependent(self) -> bool:
        return self.lam_t_lookup is not None

    def lam_for(self, stock: str) -> float:
        return float(self.lam_lookup.get(stock, float("nan")))

"""Optimal trading strategies, model-aware.

We expose three strategies:

  * :func:`ow_optimal_strategy`   — OW (linear) closed form.
  * :func:`afs_optimal_strategy`  — AFS (sqrt) variant. The target-position
                                    rule is the same as OW because the AFS
                                    closed-form for the linear-quadratic
                                    objective coincides with OW when impact
                                    cost is computed instantaneously per
                                    project.ipynb. The downstream impact /
                                    P&L decomposition uses the sqrt
                                    nonlinearity through the simulator.
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
) -> np.ndarray:
    """OW-optimal intraday execution.

    Target position:   X*_t = α_t · ADV / (2 λ σ),  clipped to ±`max_position_adv`·ADV.
    Position update:   q_t = κ · (X*_t − X_{t-1}),  κ = β = ln 2 / H_bins.
    Position is ramped down linearly over the final `liquidation_minutes`.
    """
    alpha = np.asarray(alpha, dtype=float)
    # The synthetic alpha leaves NaN at the last h_bins of each day (no
    # fwd_ret defined); treat those as "no view" so they do not poison
    # the cumulative position via NaN propagation.
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    n = alpha.shape[0]
    if lam <= 0 or n == 0:
        return np.zeros(n)

    H_bins = half_life_minutes * BINS_PER_MINUTE
    beta = np.log(2.0) / H_bins
    kappa = beta

    target = alpha * adv / (2.0 * lam * sigma)
    max_pos = max_position_adv * adv
    target = np.clip(target, -max_pos, max_pos)

    liq_bins = min(liquidation_minutes * BINS_PER_MINUTE, n)
    if liq_bins > 0:
        ramp = np.ones(n)
        ramp[n - liq_bins :] = np.linspace(1.0, 0.0, liq_bins)
        target = target * ramp

    pos = 0.0
    trades = np.zeros(n)
    for t in range(n):
        trades[t] = kappa * (target[t] - pos)
        pos += trades[t]
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
) -> np.ndarray:
    """AFS (sqrt) optimal execution.

    Per project.ipynb, the target-position / adjustment-speed rule matches
    the OW closed form. The model-specific (sqrt) nonlinearity appears in
    the impact-state recursion (q̃_t = σ · sign(q) · √(|q| / ADV)) handled
    by the simulator, not in the strategy itself.

    Kept as a separate function so that a strict AFS closed form can be
    dropped in here later without touching callers.
    """
    return ow_optimal_strategy(
        alpha,
        sigma=sigma,
        adv=adv,
        lam=lam,
        half_life_minutes=half_life_minutes,
        max_position_adv=max_position_adv,
        liquidation_minutes=liquidation_minutes,
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
        raise KeyError(
            f"Unknown strategy {name!r}; choose from {sorted(STRATEGIES)}"
        )
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

    model_type: ModelType                          # 'linear' or 'sqrt'
    half_life_minutes: float
    lam_lookup: dict[str, float]                   # stock -> scalar λ
    lam_t_lookup: dict[tuple[str, object], np.ndarray] | None = None
    strategy: str = "ow"                           # 'ow' | 'afs' | 'ext_ow'

    def is_time_dependent(self) -> bool:
        return self.lam_t_lookup is not None

    def lam_for(self, stock: str) -> float:
        return float(self.lam_lookup.get(stock, float("nan")))

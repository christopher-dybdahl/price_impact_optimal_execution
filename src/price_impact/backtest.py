"""Waelbroeck-style simulator for generic trade paths.

Implements the price construction from Part 4 of the course:

    P_unpert(t) = P_0 · (1 + cum_ret(t) + g_others(t))
    P_pert(t)   = P_0 · (1 + cum_ret(t) + g_full(t))

where g(t) is the "impact contribution":

    constant λ:   g(t) = λ · Ī(t)
    time-dep λ:   g(t) = λ(t) · Ī(t)     (extended OW; via `lam_t`)

Ī(t) is the OU recursion on normalised flow (linear or sqrt). The simulator
uses explicit flow accounting on a common bin grid:

    q_agg    = aggregate signed tape flow
    q_us     = our signed trade path
    q_others = q_agg - q_us

The "unperturbed" state is driven by q_others and the "perturbed" state by
q_agg. The simulator accepts **any** signed trade path `q_us` (not only the
OW-optimal one) and supports both `carry='daily'` (reset Ī to 0 each day) and
`carry='multi'` (carry both impact states and inventory across days with
no overnight impact decay).

This module intentionally keeps a *single* simulator entrypoint
(:func:`run_backtest`) and a small set of helpers — anything more belongs in
:mod:`results` (reporting) or :mod:`runner` (orchestration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from .impact_states import (
    ModelType,
    decay_from_half_life,
    q_tilde,
)
from .strategy import ImpactModel

CarryMode = Literal["daily", "multi"]


# ---------------------------------------------------------------------------
# Core recursion.
# ---------------------------------------------------------------------------
def ou_recursion(
    qt: np.ndarray,
    decay: float,
    i0: float = 0.0,
    *,
    no_initial_decay: bool = False,
) -> np.ndarray:
    """Ī_{t+1} = decay * Ī_t + q̃_t  with Ī_0 = i0.

    When carrying across sessions, ``no_initial_decay`` makes the first bin
    start exactly from ``i0`` rather than applying a synthetic overnight decay
    step before the first observed flow.
    """
    qt = np.asarray(qt, dtype=float)
    n = qt.shape[0]
    out = np.empty(n)
    state = float(i0)
    for t in range(n):
        state = (state if no_initial_decay and t == 0 else decay * state) + qt[t]
        out[t] = state
    return out


def waelbroeck_prices(
    mid: np.ndarray,
    q_agg: np.ndarray,
    q_us: np.ndarray,
    lam: float | np.ndarray,
    half_life_minutes: float,
    sigma: float,
    adv: float,
    model_type: ModelType = "linear",
    i_others0: float = 0.0,
    i_full0: float = 0.0,
    no_initial_decay: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate one (stock, date) session.

    Parameters
    ----------
    q_agg
        Aggregate signed flow from the tape, in shares per bin.
    q_us
        Our signed flow, in the same units and sign convention as ``q_agg``.
    lam : float or array of length len(mid)
        Scalar λ for the OW / AFS baseline, or a per-bin λ_t array for the
        extended-OW (time-dependent λ) model. The simulator multiplies
        elementwise, so the same formula works for both cases.

    Returns
    -------
    dict with keys:
        p_mid      — observed mid used to define the common baseline return
        p_unpert   — no-us Waelbroeck path, driven by q_others
        p_pert     — with-us Waelbroeck path, driven by q_agg
        i_others   — impact state Ī(q_agg - q_us)
        i_full     — impact state Ī(q_agg)
        g_others   — λ · Ī_others   (or λ_t · Ī_others for time-dep λ)
        g_full     — λ · Ī_full
        delta_g    — g_full - g_others
        position   — cumulative position from q_us (shares)
        cum_ret    — cumulative observed mid return from P0

    Backward-compatible aliases are also returned:
        p_sim=p_pert, i_ref=i_others, i_sim=i_full, g_ref=g_others,
        g_sim=g_full, q_sim=q_us.
    """
    mid = np.asarray(mid, dtype=float)
    q_agg = np.asarray(q_agg, dtype=float)
    q_us = np.asarray(q_us, dtype=float)
    if not (len(mid) == len(q_agg) == len(q_us)):
        raise ValueError("mid, q_agg, q_us must have the same length")
    n = len(mid)

    q_others = q_agg - q_us
    qt_others = q_tilde(q_others, sigma, adv, model_type)
    qt_full = q_tilde(q_agg, sigma, adv, model_type)
    decay = decay_from_half_life(half_life_minutes)

    i_others = ou_recursion(
        qt_others, decay, i_others0, no_initial_decay=no_initial_decay
    )
    i_full = ou_recursion(qt_full, decay, i_full0, no_initial_decay=no_initial_decay)

    lam_arr = np.broadcast_to(np.asarray(lam, dtype=float), (n,))
    g_others = lam_arr * i_others
    g_full = lam_arr * i_full
    delta_g = g_full - g_others

    cum_ret = (mid - mid[0]) / mid[0] if mid[0] != 0 else np.zeros(n)
    p_unpert = mid[0] * (1.0 + cum_ret + g_others)
    p_pert = mid[0] * (1.0 + cum_ret + g_full)
    position = np.cumsum(q_us)

    return {
        "p_mid": mid,
        "p_unpert": p_unpert,
        "p_pert": p_pert,
        "i_others": i_others,
        "i_full": i_full,
        "g_others": g_others,
        "g_full": g_full,
        "delta_g": delta_g,
        "q_agg": q_agg,
        "q_us": q_us,
        "q_others": q_others,
        "position": position,
        "cum_ret": cum_ret,
        # Legacy aliases used by older notebook cells / artifacts.
        "p_sim": p_pert,
        "i_ref": i_others,
        "i_sim": i_full,
        "g_ref": g_others,
        "g_sim": g_full,
        "q_sim": q_us,
    }


def mark_to_market_pnl(price: np.ndarray, position: np.ndarray) -> float:
    """Discrete P&L  ∑_t position_{t-1} · (P_t - P_{t-1}). position_0 = 0."""
    price = np.asarray(price, dtype=float)
    position = np.asarray(position, dtype=float)
    if len(price) < 2:
        return 0.0
    return float(np.sum(position[:-1] * np.diff(price)))


# ---------------------------------------------------------------------------
# Generic backtest over (stock, date) panel.
# ---------------------------------------------------------------------------
TradeProvider = Callable[
    [str, pd.Timestamp, pd.DataFrame, "BacktestContext"],
    np.ndarray,
]
"""Callable returning the simulated trade path `q_sim` for one (stock, date).

Signature: ``provider(stock, date, day_df, ctx) -> np.ndarray``
where `day_df` has the per-bin columns required by the run, and `ctx`
exposes σ, ADV, λ and the chosen model.
"""


@dataclass
class BacktestContext:
    """Per-day context handed to a trade provider so it can size positions."""

    stock: str
    date: pd.Timestamp
    sigma: float
    adv: float
    lam: float | np.ndarray
    model: ImpactModel


@dataclass
class DaySimulation:
    """One simulated trading day. Arrays are aligned to the day's bins."""

    stock: str
    date: pd.Timestamp
    time: np.ndarray
    p_mid: np.ndarray
    p_unpert: np.ndarray
    p_pert: np.ndarray
    p_sim: np.ndarray
    i_others: np.ndarray
    i_full: np.ndarray
    i_ref: np.ndarray
    i_sim: np.ndarray
    g_others: np.ndarray
    g_full: np.ndarray
    delta_g: np.ndarray
    g_ref: np.ndarray
    g_sim: np.ndarray
    position: np.ndarray
    q_agg: np.ndarray
    q_us: np.ndarray
    q_others: np.ndarray
    q_sim: np.ndarray
    volume: np.ndarray | None
    participation: np.ndarray | None
    alpha: np.ndarray | None
    fwd_ret: np.ndarray | None
    sigma: float
    adv: float
    lam: float | np.ndarray
    pnl_unpert: float  # mark-to-market on no-us Waelbroeck path
    pnl_pert: float  # mark-to-market on with-us Waelbroeck path
    pnl_mid_raw: float  # mark-to-market on observed mid
    pnl_mid: float  # legacy alias: pnl_unpert
    pnl_sim: float  # legacy alias: pnl_pert
    flow_flag_count: int = 0


@dataclass
class BacktestResult:
    """Output of a full backtest run."""

    days: list[DaySimulation]
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict = field(default_factory=dict)
    model: ImpactModel | None = None

    def to_daily(self) -> pd.DataFrame:
        if self.daily.empty and self.days:
            self.daily = pd.DataFrame(_daily_rows(self.days))
        return self.daily


def _daily_rows(days: list[DaySimulation]) -> list[dict]:
    rows: list[dict] = []
    for d in days:
        lam_max = float(np.max(np.abs(d.delta_g))) if len(d.delta_g) else 0.0
        impact_cost = d.pnl_unpert - d.pnl_pert
        day_volume = (
            float(np.nansum(np.abs(d.volume))) if d.volume is not None else float("nan")
        )
        abs_turnover = float(np.sum(np.abs(d.q_us)))
        day_participation = (
            abs_turnover / day_volume
            if np.isfinite(day_volume) and day_volume > 0
            else float("nan")
        )
        mean_participation = (
            float(np.nanmean(d.participation))
            if d.participation is not None and len(d.participation)
            else float("nan")
        )
        max_participation = (
            float(np.nanmax(d.participation))
            if d.participation is not None and len(d.participation)
            else float("nan")
        )
        rows.append(
            {
                "stock": d.stock,
                "date": d.date,
                "pnl_unpert": d.pnl_unpert,
                "pnl_pert": d.pnl_pert,
                "pnl_mid_raw": d.pnl_mid_raw,
                "pnl_mid": d.pnl_mid,
                "pnl_sim": d.pnl_sim,
                "impact_cost": impact_cost,
                "turnover": abs_turnover,
                "day_volume": day_volume,
                "day_participation": day_participation,
                "mean_bin_participation": mean_participation,
                "max_bin_participation": max_participation,
                "abs_position_mean": float(np.mean(np.abs(d.position))),
                "abs_position_max": float(np.max(np.abs(d.position))),
                "max_impact_dislocation": lam_max,
                "flow_flag_count": int(d.flow_flag_count),
                "n_bins": int(len(d.position)),
            }
        )
    return rows


def run_backtest(
    merged: pd.DataFrame,
    model: ImpactModel,
    trade_provider: TradeProvider,
    daily_stats: pd.DataFrame,
    carry: CarryMode = "daily",
    overnight_minutes: float = 16 * 60,
    q_reference_col: str = "orderFlow",
    volume_col: str = "trade",
    mid_col: str = "mid",
    time_col: str = "time",
    alpha_col: str | None = "alpha",
    fwd_ret_col: str | None = "fwd_ret",
) -> BacktestResult:
    """Run the Waelbroeck simulator over every (stock, date) in `merged`.

    Parameters
    ----------
    merged
        Long-form bin panel that contains [stock, date, time, mid, orderFlow]
        and (optionally) [trade, alpha, fwd_ret]. ``orderFlow`` is interpreted
        as q_agg and the trade provider output as q_us, with
        q_others = q_agg - q_us. The trade provider gets the per-day slice.
    model
        :class:`ImpactModel` carrying λ lookups and model_type.
    trade_provider
        Callable returning `q_sim` for each (stock, date). For an OW-optimal
        run pass :func:`make_optimal_provider(...)`.
    daily_stats
        DataFrame indexed by (stock, date) with at least `sigma` and `ADV`.
    carry
        'daily' — reset Ī and position at the start of each day.
        'multi' — carry Ī and inventory across days within a stock. The next
        session's impact state starts exactly where the previous session ended,
        as if trading were continuous.
    overnight_minutes
        Retained for API compatibility; multi-day impact carry does not apply
        overnight decay.
    """
    if carry not in ("daily", "multi"):
        raise ValueError("carry must be 'daily' or 'multi'")

    _ = overnight_minutes
    days: list[DaySimulation] = []

    df = merged.sort_values(["stock", "date", time_col]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    for stock, g_stock in df.groupby("stock", sort=False):
        lam_scalar = model.lam_for(stock)
        if not np.isfinite(lam_scalar) or lam_scalar <= 0:
            continue
        i_others_carry = 0.0
        i_full_carry = 0.0
        pos_carry = 0.0
        for date, g_day in g_stock.groupby("date", sort=False):
            g_day = g_day.sort_values(time_col)
            try:
                ds = daily_stats.loc[(stock, date)]
            except KeyError:
                continue
            sigma = float(ds["sigma"])
            adv = float(ds["ADV"])
            if not (np.isfinite(sigma) and np.isfinite(adv) and adv > 0):
                continue

            # Time-dependent λ array (if extended-OW), else scalar.
            if model.is_time_dependent():
                lam = model.lam_t_lookup.get((stock, date))
                if lam is None:
                    lam = np.full(len(g_day), lam_scalar)
            else:
                lam = lam_scalar

            ctx = BacktestContext(
                stock=stock, date=date, sigma=sigma, adv=adv, lam=lam, model=model
            )
            q_us = np.asarray(trade_provider(stock, date, g_day, ctx), dtype=float)
            if q_us.shape[0] != len(g_day):
                raise ValueError(
                    f"trade_provider returned {q_us.shape[0]} values for "
                    f"({stock!r}, {date}); expected {len(g_day)}"
                )

            mid = g_day[mid_col].to_numpy(dtype=float)
            q_agg = g_day[q_reference_col].to_numpy(dtype=float)
            volume = (
                g_day[volume_col].to_numpy(dtype=float)
                if volume_col and volume_col in g_day.columns
                else None
            )
            if volume is not None:
                denom = np.abs(volume)
                participation = np.divide(
                    np.abs(q_us),
                    denom,
                    out=np.full_like(q_us, np.nan, dtype=float),
                    where=denom > 0,
                )
            else:
                participation = None

            flow_flag = np.abs(q_us) > (np.abs(q_agg) + 1e-12)
            sim = waelbroeck_prices(
                mid=mid,
                q_agg=q_agg,
                q_us=q_us,
                lam=lam,
                half_life_minutes=model.half_life_minutes,
                sigma=sigma,
                adv=adv,
                model_type=model.model_type,
                i_others0=i_others_carry if carry == "multi" else 0.0,
                i_full0=i_full_carry if carry == "multi" else 0.0,
                no_initial_decay=carry == "multi",
            )

            position = sim["position"] + (pos_carry if carry == "multi" else 0.0)
            pnl_unpert = mark_to_market_pnl(sim["p_unpert"], position)
            pnl_pert = mark_to_market_pnl(sim["p_pert"], position)
            pnl_mid_raw = mark_to_market_pnl(mid, position)

            alpha_arr = (
                g_day[alpha_col].to_numpy(dtype=float)
                if alpha_col and alpha_col in g_day.columns
                else None
            )
            fwd_arr = (
                g_day[fwd_ret_col].to_numpy(dtype=float)
                if fwd_ret_col and fwd_ret_col in g_day.columns
                else None
            )

            days.append(
                DaySimulation(
                    stock=stock,
                    date=date,
                    time=g_day[time_col].to_numpy(),
                    p_mid=mid,
                    p_unpert=sim["p_unpert"],
                    p_pert=sim["p_pert"],
                    p_sim=sim["p_sim"],
                    i_others=sim["i_others"],
                    i_full=sim["i_full"],
                    i_ref=sim["i_ref"],
                    i_sim=sim["i_sim"],
                    g_others=sim["g_others"],
                    g_full=sim["g_full"],
                    delta_g=sim["delta_g"],
                    g_ref=sim["g_ref"],
                    g_sim=sim["g_sim"],
                    position=position,
                    q_agg=sim["q_agg"],
                    q_us=sim["q_us"],
                    q_others=sim["q_others"],
                    q_sim=q_us,
                    volume=volume,
                    participation=participation,
                    alpha=alpha_arr,
                    fwd_ret=fwd_arr,
                    sigma=sigma,
                    adv=adv,
                    lam=lam,
                    pnl_unpert=pnl_unpert,
                    pnl_pert=pnl_pert,
                    pnl_mid_raw=pnl_mid_raw,
                    pnl_mid=pnl_unpert,
                    pnl_sim=pnl_pert,
                    flow_flag_count=int(flow_flag.sum()),
                )
            )

            if carry == "multi":
                i_others_carry = float(sim["i_others"][-1])
                i_full_carry = float(sim["i_full"][-1])
                pos_carry = float(position[-1])

    result = BacktestResult(days=days, model=model)
    result.to_daily()
    return result


# ---------------------------------------------------------------------------
# Useful trade providers.
# ---------------------------------------------------------------------------
def make_optimal_provider(
    strategy_fn,
    *,
    max_position_adv: float = 0.005,
    liquidation_minutes: int = 30,
    alpha_col: str = "alpha",
    carry: str = "daily",
):
    """Return a trade provider that calls `strategy_fn` per (stock, date).

    When ``carry="multi"``, the ending normalized impact state Ī from each day
    is carried forward as ``ibar_init`` for the same stock on the next day,
    so the strategy continues from where it left off rather than resetting to 0.
    """
    ibar_carry: dict[str, float] = {}

    def provider(stock, date, day_df, ctx: BacktestContext) -> np.ndarray:
        alpha = day_df[alpha_col].to_numpy(dtype=float)
        lam_arg = (
            ctx.lam if not isinstance(ctx.lam, np.ndarray) else float(np.mean(ctx.lam))
        )
        ibar_init = ibar_carry.get(stock, 0.0) if carry == "multi" else 0.0
        trades = strategy_fn(
            alpha=alpha,
            sigma=ctx.sigma,
            adv=ctx.adv,
            lam=lam_arg,
            half_life_minutes=ctx.model.half_life_minutes,
            max_position_adv=max_position_adv,
            liquidation_minutes=liquidation_minutes,
            ibar_init=ibar_init,
        )
        if carry == "multi":
            # Compute the ending normalized impact state so the next day's
            # strategy can pick up from the right initial condition.
            decay = decay_from_half_life(ctx.model.half_life_minutes)
            qt = q_tilde(trades, ctx.sigma, ctx.adv, ctx.model.model_type)
            ibar_carry[stock] = float(ou_recursion(qt, decay, i0=ibar_init)[-1])
        return trades

    return provider


def make_fixed_provider(trades_lookup: dict[tuple[str, pd.Timestamp], np.ndarray]):
    """Provider that returns precomputed trade arrays (e.g., TWAP / VWAP)."""

    def provider(stock, date, day_df, ctx):
        key = (stock, pd.Timestamp(date))
        if key not in trades_lookup:
            return np.zeros(len(day_df))
        arr = np.asarray(trades_lookup[key], dtype=float)
        if arr.shape[0] != len(day_df):
            raise ValueError(
                f"fixed trades for {key} have length {arr.shape[0]}, "
                f"expected {len(day_df)}"
            )
        return arr

    return provider

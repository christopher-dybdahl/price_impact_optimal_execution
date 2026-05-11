"""Waelbroeck-style simulator for generic trade paths.

Implements the price construction from Part 4 of the course:

    P_sim(t) = P_0 · ( 1 + cum_ret(t) + g_sim(t) - g_ref(t) )

where g(t) is the "impact contribution":

    constant λ:   g(t) = λ · Ī(t)
    time-dep λ:   g(t) = λ(t) · Ī(t)     (extended OW; via `lam_t`)

Ī(t) is the OU recursion on normalised flow (linear or sqrt). The simulator
accepts **any** signed trade path `q_sim` (not only the OW-optimal one) and
supports both `carry='daily'` (reset Ī to 0 each day) and `carry='multi'`
(carry Ī and inventory across days with overnight decay).

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
    BINS_PER_MINUTE,
    ModelType,
    decay_from_half_life,
    overnight_decay,
    q_tilde,
)
from .strategy import ImpactModel

CarryMode = Literal["daily", "multi"]


# ---------------------------------------------------------------------------
# Core recursion.
# ---------------------------------------------------------------------------
def ou_recursion(qt: np.ndarray, decay: float, i0: float = 0.0) -> np.ndarray:
    """Ī_{t+1} = decay * Ī_t + q̃_t  with Ī_0 = i0."""
    qt = np.asarray(qt, dtype=float)
    n = qt.shape[0]
    out = np.empty(n)
    state = float(i0)
    for t in range(n):
        state = decay * state + qt[t]
        out[t] = state
    return out


def waelbroeck_prices(
    mid: np.ndarray,
    q_ref: np.ndarray,
    q_sim: np.ndarray,
    lam: float | np.ndarray,
    half_life_minutes: float,
    sigma: float,
    adv: float,
    model_type: ModelType = "linear",
    i_ref0: float = 0.0,
    i_sim0: float = 0.0,
) -> dict[str, np.ndarray]:
    """Simulate one (stock, date) session.

    Parameters
    ----------
    lam : float or array of length len(mid)
        Scalar λ for the OW / AFS baseline, or a per-bin λ_t array for the
        extended-OW (time-dependent λ) model. The simulator multiplies
        elementwise, so the same formula works for both cases.

    Returns
    -------
    dict with keys:
        p_mid    — undistorted mid (reference price)
        p_sim    — Waelbroeck-distorted simulated price
        i_ref    — reference impact state Ī^ref(t)
        i_sim    — simulated impact state Ī^sim(t)
        g_ref    — λ · Ī^ref   (or λ_t · Ī^ref for time-dep λ)
        g_sim    — λ · Ī^sim
        position — cumulative simulated position (shares)
        cum_ret  — cumulative undistorted mid return
    """
    mid = np.asarray(mid, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    q_sim = np.asarray(q_sim, dtype=float)
    if not (len(mid) == len(q_ref) == len(q_sim)):
        raise ValueError("mid, q_ref, q_sim must have the same length")
    n = len(mid)

    qt_ref = q_tilde(q_ref, sigma, adv, model_type)
    qt_sim = q_tilde(q_sim, sigma, adv, model_type)
    decay = decay_from_half_life(half_life_minutes)

    i_ref = ou_recursion(qt_ref, decay, i_ref0)
    i_sim = ou_recursion(qt_sim, decay, i_sim0)

    lam_arr = np.broadcast_to(np.asarray(lam, dtype=float), (n,))
    g_ref = lam_arr * i_ref
    g_sim = lam_arr * i_sim

    cum_ret = (mid - mid[0]) / mid[0] if mid[0] != 0 else np.zeros(n)
    p_sim = mid[0] * (1.0 + cum_ret + (g_sim - g_ref))
    position = np.cumsum(q_sim)

    return {
        "p_mid": mid,
        "p_sim": p_sim,
        "i_ref": i_ref,
        "i_sim": i_sim,
        "g_ref": g_ref,
        "g_sim": g_sim,
        "position": position,
        "cum_ret": cum_ret,
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
    p_sim: np.ndarray
    i_ref: np.ndarray
    i_sim: np.ndarray
    g_ref: np.ndarray
    g_sim: np.ndarray
    position: np.ndarray
    q_sim: np.ndarray
    alpha: np.ndarray | None
    fwd_ret: np.ndarray | None
    sigma: float
    adv: float
    lam: float | np.ndarray
    pnl_mid: float          # mark-to-market on undistorted mid
    pnl_sim: float          # mark-to-market on Waelbroeck simulated price


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
        lam_max = float(np.max(np.abs(d.g_sim))) if len(d.g_sim) else 0.0
        impact_cost = d.pnl_mid - d.pnl_sim
        rows.append(
            {
                "stock": d.stock,
                "date": d.date,
                "pnl_mid": d.pnl_mid,
                "pnl_sim": d.pnl_sim,
                "impact_cost": impact_cost,
                "turnover": float(np.sum(np.abs(d.q_sim))),
                "abs_position_mean": float(np.mean(np.abs(d.position))),
                "abs_position_max": float(np.max(np.abs(d.position))),
                "max_impact_dislocation": lam_max,
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
        and (optionally) [alpha, fwd_ret]. The trade provider gets the
        per-day slice.
    model
        :class:`ImpactModel` carrying λ lookups and model_type.
    trade_provider
        Callable returning `q_sim` for each (stock, date). For an OW-optimal
        run pass :func:`make_optimal_provider(...)`.
    daily_stats
        DataFrame indexed by (stock, date) with at least `sigma` and `ADV`.
    carry
        'daily' — reset Ī and position at the start of each day.
        'multi' — carry Ī and inventory across days within a stock, with
        overnight decay applied between sessions.
    """
    if carry not in ("daily", "multi"):
        raise ValueError("carry must be 'daily' or 'multi'")

    ovn = overnight_decay(model.half_life_minutes, overnight_minutes)
    days: list[DaySimulation] = []

    df = merged.sort_values(["stock", "date", time_col]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    for stock, g_stock in df.groupby("stock", sort=False):
        lam_scalar = model.lam_for(stock)
        if not np.isfinite(lam_scalar) or lam_scalar <= 0:
            continue
        i_ref_carry = 0.0
        i_sim_carry = 0.0
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
            q_sim = np.asarray(trade_provider(stock, date, g_day, ctx), dtype=float)
            if q_sim.shape[0] != len(g_day):
                raise ValueError(
                    f"trade_provider returned {q_sim.shape[0]} values for "
                    f"({stock!r}, {date}); expected {len(g_day)}"
                )

            mid = g_day[mid_col].to_numpy(dtype=float)
            q_ref = g_day[q_reference_col].to_numpy(dtype=float)
            sim = waelbroeck_prices(
                mid=mid,
                q_ref=q_ref,
                q_sim=q_sim,
                lam=lam,
                half_life_minutes=model.half_life_minutes,
                sigma=sigma,
                adv=adv,
                model_type=model.model_type,
                i_ref0=i_ref_carry if carry == "multi" else 0.0,
                i_sim0=i_sim_carry if carry == "multi" else 0.0,
            )

            position = sim["position"] + (pos_carry if carry == "multi" else 0.0)
            pnl_mid = mark_to_market_pnl(mid, position)
            pnl_sim = mark_to_market_pnl(sim["p_sim"], position)

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
                    p_sim=sim["p_sim"],
                    i_ref=sim["i_ref"],
                    i_sim=sim["i_sim"],
                    g_ref=sim["g_ref"],
                    g_sim=sim["g_sim"],
                    position=position,
                    q_sim=q_sim,
                    alpha=alpha_arr,
                    fwd_ret=fwd_arr,
                    sigma=sigma,
                    adv=adv,
                    lam=lam,
                    pnl_mid=pnl_mid,
                    pnl_sim=pnl_sim,
                )
            )

            if carry == "multi":
                i_ref_carry = float(sim["i_ref"][-1]) * ovn
                i_sim_carry = float(sim["i_sim"][-1]) * ovn
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
):
    """Return a trade provider that calls `strategy_fn` per (stock, date)."""

    def provider(stock, date, day_df, ctx: BacktestContext) -> np.ndarray:
        alpha = day_df[alpha_col].to_numpy(dtype=float)
        # For time-dependent λ, pass the array; the simple OW strategy ignores
        # vector λ (it expects scalar) so callers using ext-OW must supply
        # their own provider.
        lam_arg = ctx.lam if not isinstance(ctx.lam, np.ndarray) else float(np.mean(ctx.lam))
        return strategy_fn(
            alpha=alpha,
            sigma=ctx.sigma,
            adv=ctx.adv,
            lam=lam_arg,
            half_life_minutes=ctx.model.half_life_minutes,
            max_position_adv=max_position_adv,
            liquidation_minutes=liquidation_minutes,
        )

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

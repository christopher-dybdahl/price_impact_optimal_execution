"""Performance metrics, TCA decomposition, and plotting.

Given a :class:`backtest.BacktestResult`, the helpers here compute:

  * daily / cumulative P&L on the no-us Waelbroeck path and the with-us
    Waelbroeck path, plus observed-mid diagnostics and drawdown;
  * a transaction-cost-analysis (TCA) breakdown that decomposes P&L into
    well-defined components and *does not* invent heuristic numbers:

        Net P&L (pert)        = mark-to-market on P_pert
        Gross P&L (unpert)    = same position path marked on P_unpert
        Self-impact cost      = Gross - Net   (cost of our flow channel)
        Predicted-α reward    = Σ α_t · q_t · P_t   (the signal we acted on)
        Realised-α reward     = Σ fwd_ret_t · X_t · P_t  (forward return × position)

    The predicted/realised α rows are only emitted when the panel carries an
    `alpha` / `fwd_ret` column (i.e. for synthetic-alpha runs).

  * matplotlib plots (cum P&L, drawdown, sample price paths, cumulative
    impact). Plot functions accept a `save_path` and otherwise just
    `plt.show()`. They never touch a global figure registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .backtest import BacktestResult


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Aggregations.
# ---------------------------------------------------------------------------
def daily_pnl(result: "BacktestResult") -> pd.DataFrame:
    """Per-date P&L (sum across stocks)."""
    df = result.to_daily()
    if df.empty:
        return df
    return df.groupby("date")[
        [
            "pnl_unpert",
            "pnl_pert",
            "pnl_mid_raw",
            "pnl_mid",
            "pnl_sim",
            "impact_cost",
            "turnover",
            "day_volume",
            "flow_flag_count",
            "max_impact_dislocation",
        ]
    ].agg(
        pnl_unpert=("pnl_unpert", "sum"),
        pnl_pert=("pnl_pert", "sum"),
        pnl_mid_raw=("pnl_mid_raw", "sum"),
        pnl_mid=("pnl_mid", "sum"),
        pnl_sim=("pnl_sim", "sum"),
        impact_cost=("impact_cost", "sum"),
        turnover=("turnover", "sum"),
        day_volume=("day_volume", "sum"),
        flow_flag_count=("flow_flag_count", "sum"),
        max_impact=("max_impact_dislocation", "max"),
    )


def cumulative_pnl(daily: pd.DataFrame, col: str = "pnl_sim") -> pd.Series:
    return daily[col].cumsum().rename(f"cum_{col}")


def drawdown(cum: pd.Series) -> pd.Series:
    running = cum.cummax()
    return (cum - running).rename("drawdown")


def sharpe(daily: pd.DataFrame, col: str = "pnl_sim") -> float:
    s = daily[col]
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float(s.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def performance_metrics(
    result: "BacktestResult", strategy_name: str = "strategy"
) -> dict:
    """Single-strategy performance summary (Sharpe, total P&L, max DD, ...)."""
    daily = daily_pnl(result)
    if daily.empty:
        return {"strategy": strategy_name, "n_days": 0}

    cum = cumulative_pnl(daily, "pnl_pert")
    dd = drawdown(cum)

    return {
        "strategy": strategy_name,
        "n_days": int(len(daily)),
        "n_stockdays": int(len(result.to_daily())),
        "total_pnl_pert": float(daily["pnl_pert"].sum()),
        "total_pnl_unpert": float(daily["pnl_unpert"].sum()),
        "total_pnl_mid_raw": float(daily["pnl_mid_raw"].sum()),
        "total_pnl_sim": float(daily["pnl_sim"].sum()),
        "total_pnl_mid": float(daily["pnl_mid"].sum()),
        "total_impact_cost": float(daily["impact_cost"].sum()),
        "mean_daily_pnl_pert": float(daily["pnl_pert"].mean()),
        "std_daily_pnl_pert": float(daily["pnl_pert"].std()),
        "mean_daily_pnl_sim": float(daily["pnl_sim"].mean()),
        "std_daily_pnl_sim": float(daily["pnl_sim"].std()),
        "sharpe_pert": sharpe(daily, "pnl_pert"),
        "sharpe_unpert": sharpe(daily, "pnl_unpert"),
        "sharpe_mid_raw": sharpe(daily, "pnl_mid_raw"),
        "sharpe_sim": sharpe(daily, "pnl_sim"),
        "sharpe_mid": sharpe(daily, "pnl_mid"),
        "max_drawdown": float(dd.min()),
        "max_impact_dislocation": float(daily["max_impact"].max()),
        "win_rate": float((daily["pnl_pert"] > 0).mean()),
        "total_turnover": float(daily["turnover"].sum()),
        "total_day_volume": float(daily["day_volume"].sum()),
        "realized_participation": float(
            daily["turnover"].sum() / daily["day_volume"].sum()
        )
        if daily["day_volume"].sum() > 0
        else float("nan"),
        "flow_flag_count": int(daily["flow_flag_count"].sum()),
    }


# ---------------------------------------------------------------------------
# TCA decomposition.
# ---------------------------------------------------------------------------
def tca_table(result: "BacktestResult") -> pd.DataFrame:
    """Per-(stock, date) TCA decomposition.

    Columns
    -------
    net_pnl, gross_pnl, impact_cost, predicted_alpha, realised_alpha,
    turnover, gross_notional, participation diagnostics.

    `predicted_alpha` and `realised_alpha` are only computed for stock-days
    that carry the `alpha` / `fwd_ret` columns; otherwise they are NaN.
    """
    rows: list[dict] = []
    for d in result.days:
        row: dict[str, float] = {
            "stock": d.stock,
            "date": d.date,
            "net_pnl": d.pnl_pert,
            "gross_pnl": d.pnl_unpert,
            "mid_raw_pnl": d.pnl_mid_raw,
            "impact_cost": d.pnl_unpert - d.pnl_pert,
            "turnover": float(np.sum(np.abs(d.q_us))),
            "gross_notional": float(np.sum(np.abs(d.q_us) * d.p_pert)),
            "day_participation": (
                float(np.sum(np.abs(d.q_us)) / np.nansum(np.abs(d.volume)))
                if d.volume is not None and np.nansum(np.abs(d.volume)) > 0
                else float("nan")
            ),
            "flow_flag_count": float(d.flow_flag_count),
            "max_delta_g": float(np.max(np.abs(d.delta_g))) if len(d.delta_g) else 0.0,
        }
        # Predicted-alpha reward: expected return implied by α times trade × price.
        if d.alpha is not None and len(d.alpha) == len(d.q_us):
            mask = np.isfinite(d.alpha) & np.isfinite(d.q_us)
            if mask.any():
                row["predicted_alpha"] = float(
                    np.sum(d.alpha[mask] * d.q_us[mask] * d.p_unpert[mask])
                )
            else:
                row["predicted_alpha"] = float("nan")
        else:
            row["predicted_alpha"] = float("nan")
        # Realised-alpha reward: forward return times held position × price.
        if d.fwd_ret is not None and len(d.fwd_ret) == len(d.position):
            mask = np.isfinite(d.fwd_ret)
            if mask.any():
                row["realised_alpha"] = float(
                    np.sum(d.fwd_ret[mask] * d.position[mask] * d.p_unpert[mask])
                )
            else:
                row["realised_alpha"] = float("nan")
        else:
            row["realised_alpha"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def tca_summary(tca: pd.DataFrame) -> pd.Series:
    """Aggregate the per-stock-day TCA into a single summary row."""
    if tca.empty:
        return pd.Series(dtype=float)
    cols = [
        "net_pnl",
        "gross_pnl",
        "mid_raw_pnl",
        "impact_cost",
        "predicted_alpha",
        "realised_alpha",
        "turnover",
        "gross_notional",
        "flow_flag_count",
    ]
    return tca[cols].sum(min_count=1)


# ---------------------------------------------------------------------------
# Plotting helpers.
# ---------------------------------------------------------------------------
def _save_or_show(fig, save_path: Path | str | None, *, dpi: int = 120):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        import matplotlib.pyplot as plt  # noqa: F401  (re-export for ipykernel)
    return fig


def plot_cumulative_pnl(
    result: "BacktestResult",
    *,
    save_path: Path | str | None = None,
    label: str = "strategy",
):
    import matplotlib.pyplot as plt

    daily = daily_pnl(result)
    cum_pert = cumulative_pnl(daily, "pnl_pert")
    cum_unpert = cumulative_pnl(daily, "pnl_unpert")

    # Underscore-prefixed labels are silently skipped by matplotlib's legend
    # heuristic, so build the label with "net (..) / gross (..)" up front.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cum_pert.index, cum_pert.values, label=f"net (pert) [{label}]", lw=1.5)
    ax.plot(
        cum_unpert.index,
        cum_unpert.values,
        label=f"gross (unpert) [{label}]",
        lw=1.2,
        ls="--",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("Cumulative P&L (perturbed vs unperturbed)")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_drawdown(
    result: "BacktestResult",
    *,
    save_path: Path | str | None = None,
    label: str = "strategy",
):
    import matplotlib.pyplot as plt

    daily = daily_pnl(result)
    cum = cumulative_pnl(daily, "pnl_pert")
    dd = drawdown(cum)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4)
    ax.set_ylabel("Drawdown ($)")
    ax.set_title(f"Daily drawdown ({label})")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_sample_price_paths(
    result: "BacktestResult",
    stock: str,
    date: pd.Timestamp | str,
    *,
    save_path: Path | str | None = None,
):
    """Plot observed mid plus no-us and with-us Waelbroeck paths for one day."""
    import matplotlib.pyplot as plt

    date_ts = pd.Timestamp(date)
    sims = [
        d for d in result.days if d.stock == stock and pd.Timestamp(d.date) == date_ts
    ]
    if not sims:
        raise KeyError(f"No simulation for ({stock!r}, {date})")
    d = sims[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(d.p_mid, label="P_mid (observed baseline)", lw=1.0)
    axes[0].plot(d.p_unpert, label="P_unpert (others only)", lw=1.0, ls=":")
    axes[0].plot(d.p_pert, label="P_pert (full aggregate)", lw=1.0, ls="--")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].set_title(f"{stock} {date_ts.date()}  —  price paths")

    axes[1].plot(d.g_full, label=r"$\lambda \cdot \bar I^{full}$", lw=1.0)
    axes[1].plot(d.g_others, label=r"$\lambda \cdot \bar I^{others}$", lw=1.0, ls=":")
    axes[1].plot(d.delta_g, label=r"$\Delta g$ (us)", lw=1.0, ls="--")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("Cumulative impact (rel.)")
    axes[1].legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_cumulative_impact(
    result: "BacktestResult",
    *,
    save_path: Path | str | None = None,
):
    """Cross-stock distribution of max(|Δg|) per day (our impact dislocation)."""
    import matplotlib.pyplot as plt

    daily = result.to_daily()
    if daily.empty:
        raise RuntimeError("Empty result")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        [
            daily[daily["stock"] == s]["max_impact_dislocation"].values
            for s in sorted(daily["stock"].unique())
        ],
        labels=sorted(daily["stock"].unique()),
        showfliers=False,
    )
    ax.set_ylabel(r"max $|\Delta g|$ per day")
    ax.set_title("With-us vs no-us impact dislocation by stock")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig

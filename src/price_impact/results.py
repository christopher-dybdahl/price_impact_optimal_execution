"""Performance metrics, TCA decomposition, and plotting.

Given a :class:`backtest.BacktestResult`, the helpers here compute:

  * daily / cumulative P&L on both the undistorted mid and the Waelbroeck
    simulated price, plus drawdown;
  * a transaction-cost-analysis (TCA) breakdown that decomposes P&L into
    well-defined components and *does not* invent heuristic numbers:

        Net P&L (sim)         = mark-to-market on Waelbroeck-distorted price
        Gross P&L (mid)       = mark-to-market on undistorted mid (alpha capture)
        Self-impact cost      = Gross - Net   (cost of pushing the price)
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
    from .backtest import BacktestResult, DaySimulation


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Aggregations.
# ---------------------------------------------------------------------------
def daily_pnl(result: "BacktestResult") -> pd.DataFrame:
    """Per-date P&L (sum across stocks)."""
    df = result.to_daily()
    if df.empty:
        return df
    return (
        df.groupby("date")[
            ["pnl_mid", "pnl_sim", "impact_cost", "turnover", "max_impact_dislocation"]
        ]
        .agg(
            pnl_mid=("pnl_mid", "sum"),
            pnl_sim=("pnl_sim", "sum"),
            impact_cost=("impact_cost", "sum"),
            turnover=("turnover", "sum"),
            max_impact=("max_impact_dislocation", "max"),
        )
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


def performance_metrics(result: "BacktestResult", strategy_name: str = "strategy") -> dict:
    """Single-strategy performance summary (Sharpe, total P&L, max DD, ...)."""
    daily = daily_pnl(result)
    if daily.empty:
        return {"strategy": strategy_name, "n_days": 0}

    cum = cumulative_pnl(daily, "pnl_sim")
    dd = drawdown(cum)

    return {
        "strategy": strategy_name,
        "n_days": int(len(daily)),
        "n_stockdays": int(len(result.to_daily())),
        "total_pnl_sim": float(daily["pnl_sim"].sum()),
        "total_pnl_mid": float(daily["pnl_mid"].sum()),
        "total_impact_cost": float(daily["impact_cost"].sum()),
        "mean_daily_pnl_sim": float(daily["pnl_sim"].mean()),
        "std_daily_pnl_sim": float(daily["pnl_sim"].std()),
        "sharpe_sim": sharpe(daily, "pnl_sim"),
        "sharpe_mid": sharpe(daily, "pnl_mid"),
        "max_drawdown": float(dd.min()),
        "max_impact_dislocation": float(daily["max_impact"].max()),
        "win_rate": float((daily["pnl_sim"] > 0).mean()),
        "total_turnover": float(daily["turnover"].sum()),
    }


# ---------------------------------------------------------------------------
# TCA decomposition.
# ---------------------------------------------------------------------------
def tca_table(result: "BacktestResult") -> pd.DataFrame:
    """Per-(stock, date) TCA decomposition.

    Columns
    -------
    net_pnl, gross_pnl, impact_cost, predicted_alpha, realised_alpha,
    turnover, gross_notional.

    `predicted_alpha` and `realised_alpha` are only computed for stock-days
    that carry the `alpha` / `fwd_ret` columns; otherwise they are NaN.
    """
    rows: list[dict] = []
    for d in result.days:
        row: dict[str, float] = {
            "stock": d.stock,
            "date": d.date,
            "net_pnl": d.pnl_sim,
            "gross_pnl": d.pnl_mid,
            "impact_cost": d.pnl_mid - d.pnl_sim,
            "turnover": float(np.sum(np.abs(d.q_sim))),
            "gross_notional": float(np.sum(np.abs(d.q_sim) * d.p_mid)),
        }
        # Predicted-alpha reward: expected return implied by α times trade × price.
        if d.alpha is not None and len(d.alpha) == len(d.q_sim):
            mask = np.isfinite(d.alpha) & np.isfinite(d.q_sim)
            if mask.any():
                row["predicted_alpha"] = float(
                    np.sum(d.alpha[mask] * d.q_sim[mask] * d.p_mid[mask])
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
                    np.sum(d.fwd_ret[mask] * d.position[mask] * d.p_mid[mask])
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
        "impact_cost",
        "predicted_alpha",
        "realised_alpha",
        "turnover",
        "gross_notional",
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
    cum_sim = cumulative_pnl(daily, "pnl_sim")
    cum_mid = cumulative_pnl(daily, "pnl_mid")

    # Underscore-prefixed labels are silently skipped by matplotlib's legend
    # heuristic, so build the label with "net (..) / gross (..)" up front.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cum_sim.index, cum_sim.values, label=f"net (sim) [{label}]", lw=1.5)
    ax.plot(
        cum_mid.index, cum_mid.values, label=f"gross (mid) [{label}]", lw=1.2, ls="--",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("Cumulative P&L (net vs gross)")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_drawdown(result: "BacktestResult", *, save_path: Path | str | None = None,
                  label: str = "strategy"):
    import matplotlib.pyplot as plt

    daily = daily_pnl(result)
    cum = cumulative_pnl(daily, "pnl_sim")
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
    """Plot the undistorted mid and the Waelbroeck-distorted path for one day."""
    import matplotlib.pyplot as plt

    date_ts = pd.Timestamp(date)
    sims = [d for d in result.days if d.stock == stock and pd.Timestamp(d.date) == date_ts]
    if not sims:
        raise KeyError(f"No simulation for ({stock!r}, {date})")
    d = sims[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(d.p_mid, label="P_mid (undistorted)", lw=1.0)
    axes[0].plot(d.p_sim, label="P_sim (Waelbroeck)", lw=1.0, ls="--")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].set_title(f"{stock} {date_ts.date()}  —  price paths")

    axes[1].plot(d.g_sim, label=r"$\lambda \cdot \bar I^{sim}$", lw=1.0)
    axes[1].plot(d.g_ref, label=r"$\lambda \cdot \bar I^{ref}$", lw=1.0, ls=":")
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
    """Cross-stock distribution of max(|λ Ī^sim|) per day (impact dislocation)."""
    import matplotlib.pyplot as plt

    daily = result.to_daily()
    if daily.empty:
        raise RuntimeError("Empty result")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        [daily[daily["stock"] == s]["max_impact_dislocation"].values
         for s in sorted(daily["stock"].unique())],
        labels=sorted(daily["stock"].unique()),
        showfliers=False,
    )
    ax.set_ylabel(r"max $|\lambda \bar I^{sim}|$ per day")
    ax.set_title("Impact dislocation by stock")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig

"""High-level backtest orchestration + artifact saving.

The notebook should only need two things from here:

    from price_impact.runner import BacktestConfig, run_and_save

    cfg = BacktestConfig(
        name='ow_daily_h60', model_type='linear', carry='daily',
        half_life_minutes=60, rho=0.05, tau_bins=180,
    )
    result = run_and_save(panel, alphas, lam_lookup, cfg)

This builds the merged bin panel, runs the Waelbroeck simulator under the
chosen carry mode, computes metrics + TCA, saves all plots and tables to
``saved/<cfg.name>/``, and returns the :class:`backtest.BacktestResult`
along with the summary tables.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from . import results as rep
from .backtest import (
    BacktestResult,
    make_fixed_provider,
    make_optimal_provider,
    run_backtest,
)
from .impact_states import ModelType
from .strategy import ImpactModel, afs_optimal_strategy, ow_optimal_strategy

CarryMode = Literal["daily", "multi"]


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """All knobs for one backtest run. Pass to :func:`run_and_save`."""

    name: str
    model_type: ModelType = "linear"           # 'linear' (OW) or 'sqrt' (AFS)
    strategy: str = "ow"                       # 'ow' | 'afs' | 'ext_ow'
    carry: CarryMode = "daily"                 # 'daily' | 'multi'
    half_life_minutes: float = 60.0
    tau_bins: int = 180                        # τ explanation horizon
    rho: float = 0.05                          # synthetic-alpha correlation target
    h_alpha_bins: int = 1                      # α horizon
    max_position_adv: float = 0.005
    liquidation_minutes: int = 30
    overnight_minutes: float = 16 * 60
    seed: int = 42
    save_root: Path | str = field(default_factory=lambda: Path("saved"))

    def save_dir(self) -> Path:
        return Path(self.save_root) / self.name


# ---------------------------------------------------------------------------
# Strategy/model wiring.
# ---------------------------------------------------------------------------
def _strategy_fn_for(name: str):
    n = name.lower()
    if n in ("ow", "linear"):
        return ow_optimal_strategy
    if n in ("afs", "sqrt"):
        return afs_optimal_strategy
    raise ValueError(
        f"strategy {name!r} not supported by run_and_save; for ext_ow you must "
        "pass a custom trade_provider to run_backtest directly."
    )


def build_impact_model(
    cfg: BacktestConfig,
    lam_lookup: dict[str, float] | pd.Series,
    lam_t_lookup: dict | None = None,
) -> ImpactModel:
    if isinstance(lam_lookup, pd.Series):
        lam_lookup = lam_lookup.to_dict()
    return ImpactModel(
        model_type=cfg.model_type,
        half_life_minutes=cfg.half_life_minutes,
        lam_lookup=lam_lookup,
        lam_t_lookup=lam_t_lookup,
        strategy=cfg.strategy,
    )


# ---------------------------------------------------------------------------
# Panel construction.
# ---------------------------------------------------------------------------
def build_merged_panel(
    bins: pd.DataFrame,
    alphas: pd.DataFrame | None = None,
    daily_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge bins with alpha and (optionally) daily stats. The simulator
    consumes σ / ADV from `daily_stats` separately; merging is only for
    column access inside the trade provider."""
    cols = ["stock", "date", "time", "mid", "orderFlow"]
    if "trade" in bins.columns:
        cols.append("trade")
    df = bins[cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    if alphas is not None:
        a = alphas[["stock", "date", "time", "alpha", "fwd_ret"]].copy()
        a["date"] = pd.to_datetime(a["date"])
        df = df.merge(a, on=["stock", "date", "time"], how="left")
        # End-of-day bins have no fwd_ret (and hence no alpha) because the
        # synthetic alpha shift drops the final h_bins rows per day. Fill
        # with 0 ("no view") so they propagate as zero trades rather than
        # NaN-poisoning the cumulative position via cumsum.
        df["alpha"] = df["alpha"].fillna(0.0)
        df["fwd_ret"] = df["fwd_ret"].fillna(0.0)
    if daily_stats is not None:
        df = df.merge(
            daily_stats[["sigma", "ADV"]].reset_index(), on=["stock", "date"], how="left"
        )
    return df


# ---------------------------------------------------------------------------
# Single backtest run.
# ---------------------------------------------------------------------------
def run_one(
    merged: pd.DataFrame,
    daily_stats: pd.DataFrame,
    model: ImpactModel,
    cfg: BacktestConfig,
    trade_provider=None,
) -> BacktestResult:
    """Run a single backtest. If `trade_provider` is None, the OW/AFS optimal
    provider is wired from `cfg.strategy`."""
    if trade_provider is None:
        trade_provider = make_optimal_provider(
            _strategy_fn_for(cfg.strategy),
            max_position_adv=cfg.max_position_adv,
            liquidation_minutes=cfg.liquidation_minutes,
        )
    return run_backtest(
        merged=merged,
        model=model,
        trade_provider=trade_provider,
        daily_stats=daily_stats,
        carry=cfg.carry,
        overnight_minutes=cfg.overnight_minutes,
    )


# ---------------------------------------------------------------------------
# Save artifacts.
# ---------------------------------------------------------------------------
def save_artifacts(
    result: BacktestResult,
    cfg: BacktestConfig,
    *,
    save_sample_paths: tuple[str, pd.Timestamp] | None = None,
) -> dict[str, Path]:
    """Persist the run's plots and tables under ``saved/<cfg.name>/``."""
    save_dir = cfg.save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    daily = rep.daily_pnl(result)
    metrics = rep.performance_metrics(result, strategy_name=cfg.name)
    tca = rep.tca_table(result)
    tca_summary = rep.tca_summary(tca)

    # Tables.
    daily.to_csv(save_dir / "daily_pnl.csv")
    result.to_daily().to_csv(save_dir / "stock_day_pnl.csv", index=False)
    tca.to_csv(save_dir / "tca_stock_day.csv", index=False)
    tca_summary.to_frame(name="total").to_csv(save_dir / "tca_summary.csv")
    pd.Series(metrics).to_frame(name="value").to_csv(save_dir / "metrics.csv")

    # Config snapshot.
    cfg_dict = asdict(cfg)
    cfg_dict["save_root"] = str(cfg_dict["save_root"])
    (save_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))

    # Plots.
    rep.plot_cumulative_pnl(result, save_path=save_dir / "cum_pnl.png", label=cfg.name)
    rep.plot_drawdown(result, save_path=save_dir / "drawdown.png", label=cfg.name)
    rep.plot_cumulative_impact(result, save_path=save_dir / "impact_dislocation.png")
    sample_price_paths_plot = None
    if save_sample_paths is not None:
        stock, date = save_sample_paths
        sample_price_paths_plot = save_dir / f"price_paths_{stock}_{pd.Timestamp(date).date()}.png"
        try:
            rep.plot_sample_price_paths(
                result,
                stock=stock,
                date=date,
                save_path=sample_price_paths_plot,
            )
        except KeyError:
            sample_price_paths_plot = None
            pass

    paths = {
        "dir": save_dir,
        "daily_pnl": save_dir / "daily_pnl.csv",
        "stock_day_pnl": save_dir / "stock_day_pnl.csv",
        "tca": save_dir / "tca_stock_day.csv",
        "tca_summary": save_dir / "tca_summary.csv",
        "metrics": save_dir / "metrics.csv",
        "config": save_dir / "config.json",
        "cum_pnl_plot": save_dir / "cum_pnl.png",
        "drawdown_plot": save_dir / "drawdown.png",
        "impact_dislocation_plot": save_dir / "impact_dislocation.png",
    }
    if sample_price_paths_plot is not None:
        paths["sample_price_paths_plot"] = sample_price_paths_plot
    return paths


# ---------------------------------------------------------------------------
# Public one-shot wrapper.
# ---------------------------------------------------------------------------
@dataclass
class RunOutput:
    config: BacktestConfig
    result: BacktestResult
    daily: pd.DataFrame
    metrics: dict
    tca: pd.DataFrame
    tca_summary: pd.Series
    paths: dict[str, Path]


def run_and_save(
    bins: pd.DataFrame,
    daily_stats: pd.DataFrame,
    alphas: pd.DataFrame,
    lam_lookup: dict[str, float] | pd.Series,
    cfg: BacktestConfig,
    *,
    lam_t_lookup: dict | None = None,
    trade_provider=None,
    sample_path: tuple[str, pd.Timestamp] | None = None,
) -> RunOutput:
    """End-to-end: merge → simulate → metrics → TCA → save → return."""
    merged = build_merged_panel(bins, alphas=alphas, daily_stats=daily_stats)
    model = build_impact_model(cfg, lam_lookup, lam_t_lookup=lam_t_lookup)
    result = run_one(merged, daily_stats, model, cfg, trade_provider=trade_provider)
    paths = save_artifacts(result, cfg, save_sample_paths=sample_path)
    return RunOutput(
        config=cfg,
        result=result,
        daily=rep.daily_pnl(result),
        metrics=rep.performance_metrics(result, strategy_name=cfg.name),
        tca=rep.tca_table(result),
        tca_summary=rep.tca_summary(rep.tca_table(result)),
        paths=paths,
    )

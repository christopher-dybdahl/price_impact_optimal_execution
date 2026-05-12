"""price_impact — modular implementation of the final-project pipeline.

Public surface (re-exports the most-used names so the notebook stays light):

  Data
    load_bins, select_top_stocks, compute_daily_stats, build_panel, PanelData
  Impact states
    compute_impact_states, decay_from_half_life, overnight_decay, q_tilde
  Fitting
    build_regression_features, daily_sufficient_stats, rolling_baseline,
    half_life_grid_search, rolling_nonparametric, per_stock_lambda
  Alpha
    create_synthetic_alpha
  Strategy
    ow_optimal_strategy, afs_optimal_strategy,
    ext_ow_optimal_strategy_timedep_lambda, ImpactModel
  Backtest
    run_backtest, waelbroeck_prices, make_optimal_provider,
    make_fixed_provider, BacktestResult, DaySimulation. The simulator uses
    q_agg / q_us / q_others accounting and returns P_unpert / P_pert paths.
  Results
    performance_metrics, tca_table, tca_summary, daily_pnl, cumulative_pnl,
    drawdown, sharpe, plot_cumulative_pnl, plot_drawdown,
    plot_sample_price_paths, plot_cumulative_impact
  Runner
    BacktestConfig, RunOutput, run_and_save, build_impact_model
"""

from .alpha import create_synthetic_alpha
from .backtest import (
    BacktestResult,
    DaySimulation,
    make_fixed_provider,
    make_optimal_provider,
    mark_to_market_pnl,
    run_backtest,
    waelbroeck_prices,
)
from .data import (
    PanelData,
    build_panel,
    compute_daily_stats,
    compute_volume_curves,
    load_bins,
    select_top_stocks,
)
from .fitting import (
    build_regression_features,
    concavity_grid_search,
    daily_sufficient_stats,
    half_life_grid_search,
    per_stock_lambda,
    per_stock_lambda_stats,
    rolling_baseline,
    rolling_nonparametric,
)
from .impact_states import (
    compute_impact_states,
    compute_impact_states_concave,
    decay_from_half_life,
    overnight_decay,
    q_tilde,
    select_i_bar_column,
)
from .results import (
    cumulative_pnl,
    daily_pnl,
    drawdown,
    performance_metrics,
    plot_cumulative_impact,
    plot_cumulative_pnl,
    plot_drawdown,
    plot_sample_price_paths,
    sharpe,
    tca_summary,
    tca_table,
)
from .runner import BacktestConfig, RunOutput, build_impact_model, run_and_save
from .strategy import (
    ImpactModel,
    afs_optimal_strategy,
    ext_ow_optimal_strategy_timedep_lambda,
    get_strategy,
    ow_optimal_strategy,
)

__all__ = [
    # data
    "PanelData",
    "build_panel",
    "compute_daily_stats",
    "compute_volume_curves",
    "load_bins",
    "select_top_stocks",
    # impact states
    "compute_impact_states",
    "compute_impact_states_concave",
    "decay_from_half_life",
    "overnight_decay",
    "q_tilde",
    "select_i_bar_column",
    # fitting
    "build_regression_features",
    "concavity_grid_search",
    "daily_sufficient_stats",
    "half_life_grid_search",
    "per_stock_lambda",
    "per_stock_lambda_stats",
    "rolling_baseline",
    "rolling_nonparametric",
    # alpha
    "create_synthetic_alpha",
    # strategy
    "ImpactModel",
    "afs_optimal_strategy",
    "ext_ow_optimal_strategy_timedep_lambda",
    "get_strategy",
    "ow_optimal_strategy",
    # backtest
    "BacktestResult",
    "DaySimulation",
    "make_fixed_provider",
    "make_optimal_provider",
    "mark_to_market_pnl",
    "run_backtest",
    "waelbroeck_prices",
    # results
    "cumulative_pnl",
    "daily_pnl",
    "drawdown",
    "performance_metrics",
    "plot_cumulative_impact",
    "plot_cumulative_pnl",
    "plot_drawdown",
    "plot_sample_price_paths",
    "sharpe",
    "tca_summary",
    "tca_table",
    # runner
    "BacktestConfig",
    "RunOutput",
    "build_impact_model",
    "run_and_save",
]

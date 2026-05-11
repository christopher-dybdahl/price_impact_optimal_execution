"""Re-run only the 4 headline backtests after the NaN-alpha fix.

Rebuilds the panel + daily stats + impact state + rolling baseline λ from
scratch (~2 min), then runs the 4 configs (OW/AFS × daily/multi) and
overwrites saved/<name>/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import price_impact as pi  # noqa: E402

H_STAR = 60.0
TAU_BINS = 180
RHO = 0.05


def banner(msg, t0):
    print(f"[{time.time() - t0:5.1f}s]  {msg}")


def main():
    t0 = time.time()
    banner("loading panel (top-20 by abs orderFlow, 2019)...", t0)
    panel = pi.build_panel(ROOT.parent / "data", year=2019, top_n=20)
    banner(f"  {panel.n_stocks} stocks, {len(panel.bins):,} bins", t0)

    banner("synthetic alpha (rho=0.05)...", t0)
    alphas = pi.create_synthetic_alpha(panel.bins, rho=RHO, h_bins=1, seed=42)

    banner("impact panels at H*=60 min (both model_types)...", t0)
    impact_panels = {
        mt: pi.compute_impact_states(panel.bins, panel.daily_stats,
                                      H_STAR, model_type=mt)
        for mt in ("linear", "sqrt")
    }

    banner("rolling baseline (daily-reset and multi-day per model)...", t0)
    lam_lookups = {"daily": {}, "multi": {}}
    for carry in ("daily", "multi"):
        for mt in ("linear", "sqrt"):
            feats = pi.build_regression_features(
                impact_panels[mt], panel.bins, tau_bins=TAU_BINS, carry=carry
            )
            stats = pi.daily_sufficient_stats(feats)
            base = pi.rolling_baseline(stats, n_windows=10, offset=2)
            lam_lookups[carry][mt] = pi.per_stock_lambda(base).to_dict()
            banner(f"  fit {mt}/{carry}: {len(lam_lookups[carry][mt])} stocks", t0)

    sample_day = (panel.stocks[0], alphas["date"].iloc[len(alphas) // 2])

    configs = [
        pi.BacktestConfig(name="ow_daily",  model_type="linear", strategy="ow",
                          carry="daily", half_life_minutes=H_STAR, tau_bins=TAU_BINS,
                          rho=RHO, save_root=ROOT / "saved"),
        pi.BacktestConfig(name="ow_multi",  model_type="linear", strategy="ow",
                          carry="multi", half_life_minutes=H_STAR, tau_bins=TAU_BINS,
                          rho=RHO, save_root=ROOT / "saved"),
        pi.BacktestConfig(name="afs_daily", model_type="sqrt", strategy="afs",
                          carry="daily", half_life_minutes=H_STAR, tau_bins=TAU_BINS,
                          rho=RHO, save_root=ROOT / "saved"),
        pi.BacktestConfig(name="afs_multi", model_type="sqrt", strategy="afs",
                          carry="multi", half_life_minutes=H_STAR, tau_bins=TAU_BINS,
                          rho=RHO, save_root=ROOT / "saved"),
    ]
    runs = {}
    for cfg in configs:
        banner(f"running {cfg.name} ...", t0)
        lam = lam_lookups[cfg.carry][cfg.model_type]
        out = pi.run_and_save(panel.bins, panel.daily_stats, alphas, lam, cfg,
                              sample_path=sample_day)
        runs[cfg.name] = out
        m = out.metrics
        banner(
            f"  Sharpe(sim)={m['sharpe_sim']:+.3f}  "
            f"net=${m['total_pnl_sim']:>13,.0f}  "
            f"impact=${m['total_impact_cost']:>13,.0f}  "
            f"max|λI|={m['max_impact_dislocation']:.4f}",
            t0,
        )
    banner("done.", t0)
    return runs


if __name__ == "__main__":
    main()

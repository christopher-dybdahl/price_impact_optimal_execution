# Price Impact and Optimal Execution

An end-to-end price-impact research pipeline: impact-model fitting,
backtesting with impact-aware fills, synthetic alpha generation, and
optimal-execution strategies, built for the Quantitative Trading and Price
Impact final project of the MSc in Mathematics and Finance at Imperial
College London, with Arthur Nageleisen and Anran Severac.

Full write-up in [`final_report.pdf`](final_report.pdf); slides in
[`slides.pdf`](slides.pdf).

## What it does

The panel is US equities on a 10-second intraday grid. The pipeline runs the
whole research loop:

1. **Impact-model fitting** (`src/price_impact/fitting.py`): parametric
   **Obizhaeva–Wang (OW)** and **AFS**-type propagator models, plus a
   **non-parametric binned impact estimator with cross-stock shrinkage**,
   its shrinkage intensity tuned on a held-out month and stress-tested
   across sample-size regimes.
2. **Backtesting** (`src/price_impact/backtest.py`): a generic
   Waelbroeck-style simulator whose fills move the price through the fitted
   impact state. Two engines behind one configuration switch: a per-day
   reset baseline, and a **multi-day carry engine** that persists impact
   state and inventory across sessions with overnight decay.
3. **Optimal execution** (`src/price_impact/strategy.py`): OW/AFS
   closed-form optimal trading against **synthetic alphas**
   (`src/price_impact/alpha.py`), including misspecification experiments
   (trading with the wrong impact model) and a sensitivity analysis.

Every backtest run writes its tables and figures to `saved/<name>/` for
reproducibility; the thin orchestration notebook is
`final_pipeline.ipynb`.

## Layout

```
src/price_impact/
  data.py           # loading, universe selection, grids
  fitting.py        # OW / AFS / non-parametric binned estimator
  impact_states.py  # impact state recursions
  alpha.py          # synthetic alpha generation
  backtest.py       # Waelbroeck simulator (daily-reset and multi-day carry)
  strategy.py       # optimal execution rules
  results.py        # tables and plots
  runner.py         # experiment orchestration
final_pipeline.ipynb  # thin driver notebook
experiments/ results/ saved/ figures/ tables/
```

## Setup

```bash
poetry install
jupyter lab final_pipeline.ipynb
```

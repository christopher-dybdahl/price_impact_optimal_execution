"""Data loading and daily statistics (sigma, ADV, volume curves).

The intraday grid is 10-second bins. Input CSVs `bin{YEAR}{MM}.csv` live in
the project's `data/` directory and carry columns: stock, date, time, mid,
trade, orderFlow (and others).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BIN_SECONDS = 10
BINS_PER_MINUTE = 6


def load_bins(
    data_dir: Path | str,
    year: int | str = 2019,
    months: Iterable[int] | None = None,
    stocks: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate monthly bin files into one long DataFrame.

    Parameters
    ----------
    data_dir : Path
        Directory containing files named `bin{YEAR}{MM}.csv`.
    months : optional
        If provided, restrict to these months (1-12).
    stocks : optional
        If provided, restrict to these tickers.
    """
    data_dir = Path(data_dir)
    months = list(months) if months is not None else list(range(1, 13))

    frames: list[pd.DataFrame] = []
    for m in months:
        f = data_dir / f"bin{year}{m:02d}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if stocks is not None:
            df = df[df["stock"].isin(list(stocks))]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No bin files found in {data_dir} for {year}")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["stock", "date", "time"]).reset_index(drop=True)
    return out


def select_top_stocks(data: pd.DataFrame, top_n: int = 20) -> list[str]:
    """Top-N stocks by total absolute order-flow volume across the panel."""
    totals = (
        data.assign(abs_of=lambda x: x["orderFlow"].abs())
        .groupby("stock")["abs_of"]
        .sum()
        .sort_values(ascending=False)
    )
    return totals.head(top_n).index.tolist()


def compute_daily_stats(
    data: pd.DataFrame,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Trailing 20-day daily statistics: sigma (10-s return std) and ADV.

    Returns a DataFrame indexed by (stock, date) with columns sigma, ADV.

    sigma_{i,d}  = trailing-`lookback_days` std of 10-s mid returns
    ADV_{i,d}    = trailing-`lookback_days` mean of daily |trade| sum
    """
    df = data[["stock", "date", "time", "mid", "trade"]].copy()

    # Daily aggregates first.
    daily = (
        df.groupby(["stock", "date"], sort=False)
        .apply(
            lambda g: pd.Series(
                {
                    "ret_var": g["mid"].pct_change().var(),
                    "volume": g["trade"].abs().sum(),
                }
            )
        )
        .reset_index()
    )

    out_parts: list[pd.DataFrame] = []
    for stock, g in daily.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        g["sigma"] = (
            g["ret_var"].rolling(lookback_days, min_periods=1).mean().pipe(np.sqrt)
        )
        g["ADV"] = g["volume"].rolling(lookback_days, min_periods=1).mean()
        out_parts.append(g[["stock", "date", "sigma", "ADV"]])

    return pd.concat(out_parts, ignore_index=True).set_index(["stock", "date"])


def compute_volume_curves(
    data: pd.DataFrame, lookback_days: int = 20
) -> pd.DataFrame:
    """20-day trailing average remaining volume curve per (stock, time).

    The "curve" U(t) is the trailing fraction of daily volume that is *still
    to come* at intraday time t. Useful for VWAP scheduling.
    """
    df = data[["stock", "date", "time", "trade"]].copy()
    df["abs_trade"] = df["trade"].abs()

    # Daily total per (stock, date).
    daily_tot = df.groupby(["stock", "date"])["abs_trade"].sum().rename("day_tot")
    df = df.merge(daily_tot, on=["stock", "date"])
    df["cum_to_t"] = df.groupby(["stock", "date"])["abs_trade"].cumsum()
    df["remaining_frac"] = 1.0 - (df["cum_to_t"] - df["abs_trade"]) / df["day_tot"]

    # Trailing 20-day mean of remaining-fraction profile per (stock, time).
    out = (
        df.groupby(["stock", "time"])["remaining_frac"]
        .rolling(lookback_days, min_periods=1)
        .mean()
        .reset_index(level=[0, 1])
    )
    return out


@dataclass
class PanelData:
    """Container that bundles the bin-level panel and the daily stats."""

    bins: pd.DataFrame  # long-form bin data
    daily_stats: pd.DataFrame  # indexed by (stock, date)
    stocks: list[str]

    @property
    def n_stocks(self) -> int:
        return len(self.stocks)

    def attach_daily(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or ["sigma", "ADV"]
        return df.merge(
            self.daily_stats[cols].reset_index(), on=["stock", "date"], how="inner"
        )


def build_panel(
    data_dir: Path | str,
    year: int | str = 2019,
    top_n: int = 20,
    lookback_days: int = 20,
) -> PanelData:
    """One-shot constructor: load, restrict to top-N stocks, compute stats."""
    raw = load_bins(data_dir, year=year)
    stocks = select_top_stocks(raw, top_n=top_n)
    bins = raw[raw["stock"].isin(stocks)].copy()
    daily = compute_daily_stats(bins, lookback_days=lookback_days)
    return PanelData(bins=bins, daily_stats=daily, stocks=stocks)

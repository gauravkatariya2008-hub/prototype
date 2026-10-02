"""
Portfolio strategies for the research lab.

A strategy does one thing: on a rebalance day, score the stocks it is allowed
to see. The engine (portfolio.py) buys the top-scoring ones at the NEXT day's
open and handles all money, charges and record-keeping, so every strategy is
judged by exactly the same rules.

Every score uses only prices up to and including the rebalance day's close.
Scores are higher-is-better; NaN means "not eligible today".

None of these is a recommendation. Each is a well-known idea from investment
research, included so it can be tested honestly against simply holding the
Nifty 50 (NIFTYBEES).
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Context:
    """Full-history tables plus indicators. Every indicator looks backwards only."""
    close: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame
    _cache: dict = field(default_factory=dict)

    def _get(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def sma(self, days):
        return self._get(("sma", days), lambda: self.close.rolling(days, min_periods=days).mean())

    def returns(self):
        return self._get("ret", lambda: self.close.pct_change(fill_method=None))


@dataclass
class Strategy:
    key: str
    label: str
    description: str
    rebalance: str               # "M" month-end or "W" week-end
    top_n: int
    min_history: int             # trading days of data a stock needs before it can be picked
    score: Callable              # (ctx, i) -> Series of scores for row i
    benchmark_only: bool = False # trades the index itself rather than individual stocks


def momentum_score(ctx, i):
    """Return over the past 12 months, skipping the most recent month."""
    if i < 252:
        return pd.Series(dtype=float)
    past, recent = ctx.close.iloc[i - 252], ctx.close.iloc[i - 21]
    return recent / past - 1


def low_vol_score(ctx, i):
    """Calmer stocks score higher: minus the daily volatility of the past year."""
    if i < 252:
        return pd.Series(dtype=float)
    window = ctx.returns().iloc[i - 251:i + 1]
    vol = window.std()
    vol[window.count() < 200] = np.nan
    return -vol


def mean_reversion_score(ctx, i):
    """Biggest 5-day fall, but only for stocks still above their 200-day average."""
    if i < 200:
        return pd.Series(dtype=float)
    now, before = ctx.close.iloc[i], ctx.close.iloc[i - 5]
    change = now / before - 1
    uptrend = now > ctx.sma(200).iloc[i]
    score = -change
    score[~uptrend | (change >= 0)] = np.nan          # only stocks that actually fell
    return score


def index_trend_score(ctx, i, benchmark="NIFTYBEES"):
    """Hold the index while it is above its 200-day average; otherwise sit in cash."""
    if benchmark not in ctx.close.columns or i < 200:
        return pd.Series(dtype=float)
    above = ctx.close[benchmark].iat[i] > ctx.sma(200)[benchmark].iat[i]
    return pd.Series({benchmark: 1.0 if above else np.nan})


STRATEGIES = {s.key: s for s in [
    Strategy("momentum", "Momentum",
             "Each month, buy the stocks that rose most over the past 12 months "
             "(skipping the latest month). Hold them until they drop out of the top.",
             rebalance="M", top_n=20, min_history=260, score=momentum_score),
    Strategy("low_vol", "Low volatility",
             "Each month, hold the stocks with the calmest daily price moves over the "
             "past year. Aims to fall less in crashes rather than to win big.",
             rebalance="M", top_n=20, min_history=260, score=low_vol_score),
    Strategy("mean_reversion", "Short-term mean reversion",
             "Each week, buy the stocks that fell most over 5 days while still in a "
             "long-term uptrend, expecting a bounce. Trades often, so charges bite.",
             rebalance="W", top_n=10, min_history=210, score=mean_reversion_score),
    Strategy("index_trend", "Index trend following",
             "Hold NIFTYBEES while it is above its 200-day average; move to cash when it "
             "falls below. Aims to sidestep big crashes, usually lags in strong rallies.",
             rebalance="W", top_n=1, min_history=210, score=index_trend_score,
             benchmark_only=True),
]}

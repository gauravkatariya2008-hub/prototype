"""
Portfolio backtest engine for the research lab.

Replays a strategy day by day with real money rules:
  * decisions use prices up to a rebalance day's close; trades happen at the
    NEXT day's open (you cannot trade on a close you have not seen)
  * whole shares only; a stock you cannot afford even one share of is skipped
  * every buy and sell pays Zerodha delivery charges (costs.py), including the
    flat DP charge on each stock sold
  * liquidity is judged point-in-time: a stock must have traded enough in the
    60 days BEFORE the rebalance, not just today
  * the benchmark is simply buying NIFTYBEES on the first day and holding it

Holdings that stay in the top N are left alone (no small top-up trades, each of
which would pay a DP charge on the sell side). Stocks that drop out are sold;
new entrants are bought with an equal share of the money.

No orders are ever sent anywhere. This is arithmetic on past prices.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import costs
from data import BENCHMARK

BUY_BUFFER = 0.003            # keep this share of each buy aside for charges
MIN_TURNOVER_CR = 5.0         # Rs crore of median daily traded value, point-in-time
SPLIT_DATE = "2023-01-01"
BEAT_T = 2.0                  # monthly excess-return confidence needed to claim a real edge
BEAT_YEAR_SHARE = 0.6


@dataclass
class Result:
    strategy: str
    capital: float
    equity: pd.Series         # value of the strategy at each close
    benchmark: pd.Series      # value of buy-and-hold NIFTYBEES at each close
    trades: pd.DataFrame
    metrics: dict
    yearly: pd.DataFrame
    verdict: dict


def rebalance_days(index, freq):
    """Last trading day of each month ("M") or week ("W")."""
    s = pd.Series(index, index=index)
    if freq == "M":
        key = [index.year, index.month]
    else:
        iso = index.isocalendar()
        key = [iso.year.values, iso.week.values]
    return pd.DatetimeIndex(s.groupby(key).max().sort_values().values)


def eligible(ctx, i, strategy, min_turnover_cr, history_counts, turnover):
    """Stocks the strategy may pick on row i, using only data up to row i."""
    if strategy.benchmark_only:
        return [BENCHMARK] if BENCHMARK in ctx.close.columns else []
    row_close = ctx.close.iloc[i]
    ok = row_close.notna() & (history_counts.iloc[i] >= strategy.min_history)
    ok &= turnover.iloc[i] >= min_turnover_cr * 1e7
    if BENCHMARK in ok.index:
        ok[BENCHMARK] = False             # a stock strategy must not secretly hold the index
    return list(ok[ok].index)


def pick(ctx, i, strategy, min_turnover_cr, history_counts, turnover, top_n):
    allowed = eligible(ctx, i, strategy, min_turnover_cr, history_counts, turnover)
    if not allowed:
        return []
    scores = strategy.score(ctx, i).reindex(allowed).dropna()
    return list(scores.sort_values(ascending=False, kind="stable").index[:top_n])


def run(panel, strategy, capital=500000.0, top_n=None, start=None, end=None,
        min_turnover_cr=MIN_TURNOVER_CR, split=SPLIT_DATE):
    from strategies import Context

    top_n = top_n or strategy.top_n
    ctx = Context(panel["close"], panel["open"], panel["volume"])
    close_ff = ctx.close.ffill()
    history_counts = ctx.close.notna().cumsum()
    turnover = (ctx.close * ctx.volume).rolling(60, min_periods=40).median()

    dates = ctx.close.index
    window = dates[(dates >= pd.Timestamp(start)) if start else slice(None)]
    if end:
        window = window[window <= pd.Timestamp(end)]
    pos = {d: k for k, d in enumerate(dates)}
    # The race starts once the strategy has enough history to make a decision.
    # Starting earlier would leave it in cash while the benchmark is invested.
    rebal = {d for d in rebalance_days(window, strategy.rebalance)
             if pos[d] >= strategy.min_history}
    if not rebal:
        raise ValueError("Not enough history in the chosen period for this strategy.")
    first = min(rebal)

    cash, qty, trades = float(capital), {}, []
    pending_targets, pending_sells = None, set()
    equity, slots, unaffordable, no_price = {}, 0, 0, 0

    for d in window[window >= first]:
        i = pos[d]
        # 1. execute yesterday's decision at today's open
        if pending_targets is not None or pending_sells:
            targets = pending_targets if pending_targets is not None else []
            if pending_targets is not None:
                pending_sells |= {s for s in qty if s not in targets}
            for s in sorted(pending_sells):
                price = ctx.open[s].iat[i] if s in ctx.open else np.nan
                if np.isnan(price) or s not in qty:
                    continue                                  # no trading today; retry tomorrow
                value = qty[s] * price
                fee = costs.trade_cost(value, "sell", etf=(s == BENCHMARK))
                cash += value - fee
                trades.append({"date": d, "symbol": s, "side": "SELL", "qty": qty[s],
                               "price": price, "value": value, "charges": fee})
                del qty[s]
            pending_sells = {s for s in pending_sells if s in qty}
            if pending_targets is not None:
                new = [s for s in targets if s not in qty]
                slots += len(new)
                if new:
                    held = sum(qty[s] * close_ff[s].iat[i - 1] for s in qty) if i > 0 else 0.0
                    share = (cash + held) / len(targets)
                    budget_each = min(share, cash / len(new))
                    for s in new:
                        price = ctx.open[s].iat[i]
                        if np.isnan(price) or price <= 0:
                            no_price += 1             # not traded today (suspended, holiday)
                            continue
                        n = int(budget_each * (1 - BUY_BUFFER) // price)
                        if n <= 0:
                            unaffordable += 1
                            continue
                        value = n * price
                        fee = costs.trade_cost(value, "buy", etf=(s == BENCHMARK))
                        if value + fee > cash:
                            unaffordable += 1
                            continue
                        cash -= value + fee
                        qty[s] = n
                        trades.append({"date": d, "symbol": s, "side": "BUY", "qty": n,
                                       "price": price, "value": value, "charges": fee})
            pending_targets = None
        # 2. decide at today's close, to execute tomorrow
        if d in rebal:
            pending_targets = pick(ctx, i, strategy, min_turnover_cr,
                                   history_counts, turnover, top_n)
        # 3. mark to market
        equity[d] = cash + sum(n * close_ff[s].iat[i] for s, n in qty.items())

    equity = pd.Series(equity, name=strategy.key)
    bench = benchmark_series(ctx, close_ff, equity.index, capital)
    trades = pd.DataFrame(trades, columns=["date", "symbol", "side", "qty", "price",
                                           "value", "charges"])
    metrics = compute_metrics(equity, bench, trades, capital, slots, unaffordable)
    metrics["skipped_no_price"] = int(no_price)
    yearly = yearly_returns(equity, bench)
    verdict = judge(equity, bench, yearly, split)
    metrics.update({"split": split})
    return Result(strategy.key, capital, equity, bench, trades, metrics, yearly, verdict)


def benchmark_series(ctx, close_ff, index, capital):
    """
    Buy NIFTYBEES with all the money at the same open the strategy first trades at
    (the day after its first decision), then hold. Both start from `capital` on the
    decision day, so the race starts on the same day for both.
    """
    if BENCHMARK not in ctx.close.columns or len(index) < 2:
        return pd.Series(capital, index=index, name=BENCHMARK)
    i1 = ctx.close.index.get_loc(index[1])
    price = ctx.open[BENCHMARK].iat[i1]
    if np.isnan(price):
        price = close_ff[BENCHMARK].iat[i1]
    n = int(capital * (1 - BUY_BUFFER) // price)
    cash = capital - n * price - (costs.trade_cost(n * price, "buy", etf=True) if n else 0.0)
    values = cash + n * close_ff[BENCHMARK].reindex(index)
    values.iloc[0] = capital
    return values.rename(BENCHMARK)


def cagr(series):
    if len(series) < 2 or series.iloc[0] <= 0:
        return float("nan")
    days = (series.index[-1] - series.index[0]).days
    if days <= 0:
        return float("nan")
    return float((series.iloc[-1] / series.iloc[0]) ** (365.25 / days) - 1)


def max_drawdown(series):
    return float((series / series.cummax() - 1).min()) if len(series) else float("nan")


def compute_metrics(equity, bench, trades, capital, slots, unaffordable):
    daily = equity.pct_change(fill_method=None).dropna()
    return {
        "final_value": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / capital - 1),
        "cagr": cagr(equity),
        "max_drawdown": max_drawdown(equity),
        "volatility": float(daily.std() * np.sqrt(252)) if len(daily) > 1 else float("nan"),
        "bench_final": float(bench.iloc[-1]),
        "bench_cagr": cagr(bench),
        "bench_max_drawdown": max_drawdown(bench),
        "trades": int(len(trades)),
        "charges_paid": float(trades["charges"].sum()) if len(trades) else 0.0,
        "buy_slots": int(slots),
        "unaffordable": int(unaffordable),
        "start": equity.index[0].date().isoformat(),
        "end": equity.index[-1].date().isoformat(),
    }


def yearly_returns(equity, bench):
    """Calendar-year return of the strategy and the benchmark."""
    rows = []
    for year in sorted(set(equity.index.year)):
        e = equity[equity.index.year == year]
        b = bench[bench.index.year == year]
        prev_e = equity[equity.index.year < year]
        prev_b = bench[bench.index.year < year]
        start_e = prev_e.iloc[-1] if len(prev_e) else e.iloc[0]
        start_b = prev_b.iloc[-1] if len(prev_b) else b.iloc[0]
        rows.append({"year": int(year), "strategy": e.iloc[-1] / start_e - 1,
                     "benchmark": b.iloc[-1] / start_b - 1,
                     "partial": not len(prev_e) or year == equity.index[-1].year})
    df = pd.DataFrame(rows)
    if len(df):
        df["beat"] = df["strategy"] > df["benchmark"]
    return df


def excess_t(equity, bench):
    """Confidence that monthly returns beat the benchmark's, month by month."""
    m_e = equity.resample("ME").last().pct_change().dropna()
    m_b = bench.resample("ME").last().pct_change().dropna()
    diff = (m_e - m_b).dropna()
    if len(diff) < 12 or diff.std(ddof=1) == 0:
        return float("nan"), len(diff)
    return float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))), len(diff)


def judge(equity, bench, yearly, split=SPLIT_DATE):
    """Does the strategy genuinely beat buying and holding the index?"""
    cut = pd.Timestamp(split)
    reasons = []
    early_e, late_e = equity[equity.index < cut], equity[equity.index >= cut]
    early_b, late_b = bench[bench.index < cut], bench[bench.index >= cut]
    periods = {}
    for label, e, b in (("before", early_e, early_b), ("after", late_e, late_b)):
        periods[label] = {"cagr": cagr(e), "bench_cagr": cagr(b), "days": int(len(e))}
        if len(e) < 120:
            reasons.append(f"not enough history {label} {split} to compare")
        elif not cagr(e) > cagr(b):
            reasons.append(f"did not beat NIFTYBEES {label} {split}")
    t, months = excess_t(equity, bench)
    if not t >= BEAT_T:
        reasons.append(f"confidence {t:.1f} is below {BEAT_T:.0f} — the difference could be luck")
    full_years = yearly[~yearly["partial"]] if len(yearly) else yearly
    if len(full_years) == 0:
        reasons.append("no complete calendar year to judge")
    elif full_years["beat"].mean() < BEAT_YEAR_SHARE:
        reasons.append(f"beat NIFTYBEES in only {int(full_years['beat'].sum())} of "
                       f"{len(full_years)} full years")
    return {"beats_benchmark": not reasons, "reasons": reasons, "t": t, "months": months,
            "periods": periods,
            "years_beaten": int(full_years["beat"].sum()) if len(full_years) else 0,
            "full_years": int(len(full_years))}

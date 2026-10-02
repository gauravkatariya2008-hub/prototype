"""
Today's picks, a plan for your budget, and a hold-or-sell check on what you own.

The picks come ONLY from a strategy that beat NIFTYBEES on your own price history
(portfolio.judge). If none did, the honest answer is NIFTYBEES itself, and that is
what this module returns rather than inventing picks.

Suggestions only. Nothing here places orders.
"""

import numpy as np
import pandas as pd

import costs
import portfolio
import strategies
from data import BENCHMARK

MAX_CHARGE_SHARE = 0.01       # a position is too small if a round trip costs more than 1% of it
PICKS = 5                     # the user holds 5 stocks, so the strategies are tested with 5
HEADROOM = 1.5                # each slot gets 1.5x the minimum, so whole shares can fit


def min_position(max_share=MAX_CHARGE_SHARE):
    """Smallest position whose buy+sell charges stay under max_share of its value."""
    lo, hi = 1.0, 1e7
    for _ in range(60):
        mid = (lo + hi) / 2
        if costs.round_trip_cost(mid, mid) / mid <= max_share:
            hi = mid
        else:
            lo = mid
    return float(np.ceil(hi))


def tested_top_n(key):
    """How many stocks the picks version of a strategy holds (and is tested with)."""
    return min(strategies.STRATEGIES[key].top_n, PICKS)


def choose_strategy(results):
    """
    results: {strategy key: portfolio.Result}. Returns (key or None, table rows).
    Only strategies that beat NIFTYBEES qualify; among them, the most confident wins.
    """
    rows = []
    for key, r in results.items():
        rows.append({"key": key, "label": strategies.STRATEGIES[key].label,
                     "beats": r.verdict["beats_benchmark"], "t": r.verdict["t"],
                     "cagr": r.metrics["cagr"], "bench_cagr": r.metrics["bench_cagr"],
                     "reasons": r.verdict["reasons"]})
    passing = [r for r in rows if r["beats"]]
    best = max(passing, key=lambda r: r["t"])["key"] if passing else None
    return best, rows


def current_picks(panel, key, min_turnover_cr=portfolio.MIN_TURNOVER_CR, n=None):
    """The strategy's ranked choices on the latest day, using data up to today's close."""
    strat = strategies.STRATEGIES[key]
    ctx = strategies.Context(panel["close"], panel["open"], panel["volume"])
    history_counts = ctx.close.notna().cumsum()
    turnover = (ctx.close * ctx.volume).rolling(60, min_periods=40).median()
    i = len(ctx.close) - 1
    return portfolio.pick(ctx, i, strat, min_turnover_cr, history_counts, turnover,
                          n or strat.top_n)


def latest_prices(panel, symbols):
    close = panel["close"].ffill().iloc[-1]
    return {s: float(close[s]) for s in symbols if s in close and not np.isnan(close[s])}


def plan_budget(budget, ranked, prices, max_stocks=PICKS):
    """
    Split a budget across the ranked picks, in whole shares, keeping every position
    big enough that charges stay under 1%. Too small a budget for any stock means
    NIFTYBEES instead. Returns (rows, leftover cash, explanation).
    """
    floor = min_position()
    slots = int(min(max_stocks, budget // (floor * HEADROOM)))
    if slots == 0 or not ranked:
        return _benchmark_plan(budget, prices, floor, reason="small" if ranked else "none")

    share = budget / slots
    rows, cash = [], float(budget)
    for sym in ranked:
        if len(rows) == slots:
            break
        price = prices.get(sym)
        if not price:
            continue
        qty = int(share * (1 - portfolio.BUY_BUFFER) // price)
        cost = qty * price
        if qty == 0 or cost < floor:
            continue                                  # whole shares don't fit the size rule; next one
        fee = costs.trade_cost(cost, "buy")
        if cost + fee > cash:
            continue
        cash -= cost + fee
        rows.append({"symbol": sym, "price": price, "qty": qty, "cost": cost,
                     "buy_charges": fee, "share_of_budget": cost / budget})
    if not rows:
        return _benchmark_plan(budget, prices, floor, reason="expensive")
    note = (f"{len(rows)} stock{'s' if len(rows) > 1 else ''}, up to ₹{share:,.0f} each. Every "
            f"position is at least ₹{floor:,.0f} so that buying and selling costs under 1% of it.")
    return rows, cash, note


def _benchmark_plan(budget, prices, floor, reason):
    price = prices.get(BENCHMARK)
    why = {
        "small": f"₹{budget:,.0f} is too small to split into stocks: each position needs at "
                 f"least ₹{floor:,.0f}, or charges eat more than 1% of it.",
        "expensive": "None of the picks can be bought in whole shares with an equal share of "
                     "this budget.",
        "none": "No strategy beat NIFTYBEES on your data, so there are no stock picks to trust.",
    }[reason]
    if not price:
        return [], float(budget), why + " NIFTYBEES prices are missing, so no plan."
    qty = int(budget * (1 - portfolio.BUY_BUFFER) // price)
    if qty == 0:
        return [], float(budget), why + " The budget can't buy even one NIFTYBEES unit."
    cost = qty * price
    fee = costs.trade_cost(cost, "buy", etf=True)
    row = {"symbol": BENCHMARK, "price": price, "qty": qty, "cost": cost,
           "buy_charges": fee, "share_of_budget": cost / budget}
    return [row], float(budget - cost - fee), why + " Buying NIFTYBEES instead."


def review_holdings(holdings, prices, keep, key):
    """
    holdings: DataFrame with symbol, qty, buy_price. keep: the strategy's current
    top list (or None when no strategy passed). Returns a DataFrame with profit and
    an action based on the strategy's own rule: hold while it stays in the top list.
    """
    rows = []
    for h in holdings.itertuples(index=False):
        sym = str(h.symbol).strip().upper().removesuffix(".NS")
        if not sym or pd.isna(h.qty) or pd.isna(h.buy_price) or h.qty <= 0:
            continue
        price = prices.get(sym)
        if price is None:
            rows.append({"symbol": sym, "qty": int(h.qty), "buy_price": float(h.buy_price),
                         "price": np.nan, "value": np.nan, "profit_after_sell_charges": np.nan,
                         "action": "No price data — run python scan.py --update"})
            continue
        value = h.qty * price
        sell_fee = costs.trade_cost(value, "sell", etf=(sym == BENCHMARK))
        profit = value - h.qty * h.buy_price - sell_fee
        if sym == BENCHMARK:
            action = "HOLD — the index is the benchmark everything else must beat"
        elif keep is None:
            action = "No tested strategy supports this stock — consider NIFTYBEES"
        elif sym in keep:
            action = f"HOLD — still in {strategies.STRATEGIES[key].label}'s top list"
        else:
            action = (f"SELL at the next rebalance — dropped out of "
                      f"{strategies.STRATEGIES[key].label}'s top list")
        rows.append({"symbol": sym, "qty": int(h.qty), "buy_price": float(h.buy_price),
                     "price": price, "value": value, "profit_after_sell_charges": profit,
                     "action": action})
    return pd.DataFrame(rows)

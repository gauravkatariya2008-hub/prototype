"""
Measures how each rule ACTUALLY performed on past prices.

This is the honest core of the project. It replays every signal over years of
history and reports the win rate, average win, average loss and expectancy
after real Zerodha charges. A rule whose expectancy is negative is a rule that
lost money — the scanner will say so instead of dressing it up as a tip.

What it deliberately does NOT do is promise anything about the future.

Two biases you must keep in mind when reading the output:
  * Survivorship — price history only exists for companies still listed today,
    so delisted failures are missing and every result is flattered.
  * Sample size — a rule with 30 occurrences tells you nothing. Anything under
    MIN_SAMPLE is labelled unreliable.
  * Liquidity — stocks are filtered on how liquid they are TODAY, applied to
    their whole history. A stock that was thin five years ago still counts.

Usage:
  python evaluate.py --demo                     # synthetic data, no internet
  python evaluate.py --symbols RELIANCE,TCS     # a few real stocks
  python evaluate.py                            # whole cached universe
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import costs
import signals

MAX_HOLD = 15        # trading days; a swing trade that hasn't resolved is closed out
MIN_SAMPLE = 100     # fewer occurrences than this and the numbers are noise
STATS_FILE = Path(__file__).with_name("stats.json")


def simulate_trade(df, i, sig, max_hold=MAX_HOLD, notional=10000.0):
    """
    Replay one signal fired on bar i, entering at bar i+1. Returns None if the
    trade never triggered. Uses no data before the entry decision is made.
    """
    j0 = i + 1
    if j0 >= len(df):
        return None
    o, h = df["Open"].iat[j0], df["High"].iat[j0]
    if h < sig["trigger"]:
        return None                      # price never reached the trigger
    entry = max(sig["trigger"], o)
    if entry >= sig["target"]:
        return None                      # gapped past the target; you would not buy here
    stop, target = sig["stop"], sig["target"]
    if entry <= stop:
        return None

    outcome, exit_price, exit_idx = "timeout", df["Close"].iat[min(j0 + max_hold - 1, len(df) - 1)], None
    for j in range(j0, min(j0 + max_hold, len(df))):
        low, high = df["Low"].iat[j], df["High"].iat[j]
        hit_stop, hit_target = low <= stop, high >= target
        if hit_stop and hit_target:
            # Daily bars cannot tell which came first, so assume the bad one.
            outcome, exit_price, exit_idx = "loss", stop, j
            break
        if hit_stop:
            outcome, exit_price, exit_idx = "loss", stop, j
            break
        if hit_target:
            outcome, exit_price, exit_idx = "win", target, j
            break
    if exit_idx is None:
        if j0 + max_hold > len(df):
            outcome = "open"             # data ran out mid-trade; not a real result
        exit_idx = min(j0 + max_hold - 1, len(df) - 1)
        exit_price = df["Close"].iat[exit_idx]

    risk = entry - stop
    qty = max(1, int(notional // entry))
    gross = (exit_price - entry) * qty
    charges = costs.round_trip_cost(entry * qty, exit_price * qty)
    return {"entry_date": df.index[j0], "exit_date": df.index[exit_idx],
            "entry": entry, "exit": exit_price, "outcome": outcome,
            "r_multiple": (exit_price - entry) / risk,
            "net_rupees": gross - charges, "qty": qty,
            "held_days": exit_idx - j0 + 1, "exit_idx": exit_idx}


def collect_trades(frames, max_hold=MAX_HOLD, notional=10000.0):
    """
    Run every detector over every symbol's history. frames: {symbol: indicator df}.

    One trade at a time per stock per rule: while a trade is open, the same rule
    firing again on that stock is ignored. Otherwise a setup that fires three days
    running counts as three trades, which inflates the sample size and makes a
    weak rule look statistically solid. Trades still open when the data ends are
    dropped, because they have no result yet.
    """
    rows = []
    for symbol, df in frames.items():
        if len(df) < signals.TREND_MA + max_hold + 5:
            continue
        busy_until = {}
        for i in range(signals.TREND_MA + 2, len(df) - 1):
            for sig in signals.detect_all(df, i):
                if i <= busy_until.get(sig["signal"], -1):
                    continue
                trade = simulate_trade(df, i, sig, max_hold, notional)
                if trade is None:
                    continue
                busy_until[sig["signal"]] = trade["exit_idx"]
                if trade["outcome"] != "open":
                    rows.append({"symbol": symbol, "signal": sig["signal"], **trade})
    return pd.DataFrame(rows)


def summarize(trades):
    """Per-signal performance. Everything here is net of charges where in rupees."""
    stats = {}
    if trades.empty:
        return stats
    for name, grp in trades.groupby("signal"):
        wins = grp[grp["net_rupees"] > 0]
        losses = grp[grp["net_rupees"] <= 0]
        n = len(grp)
        stats[name] = {
            "samples": n,
            "win_rate": len(wins) / n,
            "avg_win_rupees": float(wins["net_rupees"].mean()) if len(wins) else 0.0,
            "avg_loss_rupees": float(losses["net_rupees"].mean()) if len(losses) else 0.0,
            "expectancy_rupees": float(grp["net_rupees"].mean()),
            "expectancy_r": float(grp["r_multiple"].mean()),
            "avg_held_days": float(grp["held_days"].mean()),
            "target_hit_rate": float((grp["outcome"] == "win").mean()),
            "stopped_out_rate": float((grp["outcome"] == "loss").mean()),
            "reliable": n >= MIN_SAMPLE,
        }
    return stats


def print_stats(stats, notional):
    if not stats:
        print("No trades were generated. Not enough data, or no setup ever triggered.")
        return
    print(f"\nMeasured on past data, assuming about Rs {notional:,.0f} per trade,")
    print("after Zerodha delivery charges. Sorted by expectancy (best first).\n")
    print(f"{'signal':14}{'times':>7}{'won':>7}{'avg win':>10}{'avg loss':>10}"
          f"{'per trade':>11}{'R gross':>9}  verdict")
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["expectancy_rupees"]):
        if not s["reliable"]:
            verdict = f"too few samples ({s['samples']}) — ignore"
        elif s["expectancy_rupees"] > 0:
            verdict = "made money"
        else:
            verdict = "LOST money — do not trade this"
        print(f"{name:14}{s['samples']:>7}{s['win_rate']:>7.0%}"
              f"{s['avg_win_rupees']:>10,.0f}{s['avg_loss_rupees']:>10,.0f}"
              f"{s['expectancy_rupees']:>11,.0f}{s['expectancy_r']:>9.2f}  {verdict}")
    print("\n'per trade' is rupees after charges and decides the verdict. 'R gross' is the")
    print("move in multiples of the risk, BEFORE charges — a rule can be positive in R and")
    print("still lose money once charges are paid.")
    print("\nRemember: these are past results, they flatter themselves (delisted")
    print("companies are missing from the data), and they are not a forecast.\n")


def main():
    p = argparse.ArgumentParser(description="Measure how each trading rule performed historically.")
    p.add_argument("--symbols", help="comma separated NSE symbols, e.g. RELIANCE,TCS")
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--notional", type=float, default=10000, help="rupees per trade, for cost maths")
    p.add_argument("--max-hold", type=int, default=MAX_HOLD)
    p.add_argument("--demo", action="store_true", help="synthetic prices, no internet needed")
    p.add_argument("--trades-csv", help="also write every simulated trade to this CSV")
    p.add_argument("--min-turnover", type=float, default=None,
                   help="skip stocks trading under this many Rs crore a day (default: same as scan.py)")
    args = p.parse_args()

    import data
    import universe
    min_turnover = universe.MIN_TURNOVER_CR if args.min_turnover is None else args.min_turnover
    frames = data.load_frames(symbols=args.symbols.split(",") if args.symbols else None,
                             years=args.years, demo=args.demo, min_turnover_cr=min_turnover)
    if not frames:
        raise SystemExit("No price data available. Try --demo, or run scan.py --update first.")
    print(f"Replaying {len(frames)} stocks...")
    trades = collect_trades(frames, args.max_hold, args.notional)
    stats = summarize(trades)
    print_stats(stats, args.notional)
    STATS_FILE.write_text(json.dumps({"notional": args.notional, "signals": stats}, indent=2))
    print(f"Saved to {STATS_FILE.name} — scan.py reads it to score today's picks.")
    if args.trades_csv and not trades.empty:
        trades.to_csv(args.trades_csv, index=False)
        print(f"Wrote {len(trades)} trades to {args.trades_csv}")


if __name__ == "__main__":
    main()

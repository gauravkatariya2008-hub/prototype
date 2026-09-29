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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import costs
import signals

MAX_HOLD = 15        # trading days; a swing trade that hasn't resolved is closed out
MIN_SAMPLE = 100     # fewer occurrences than this and the numbers are noise
STATS_FILE = Path(__file__).with_name("stats.json")
TRADES_FILE = Path(__file__).with_name("trades.csv")

# Stress-test standards. A rule "holds up" only if it clears all three.
SPLIT_DATE = "2023-01-01"   # compare trades before and after this date
ROBUST_T = 2.0              # day-clustered confidence must reach this
ROBUST_YEAR_SHARE = 0.6     # at least this share of years must be profitable
MIN_YEAR_TRADES = 20        # a year with fewer trades is too thin to count either way


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
            "stop": stop, "risk_per_share": risk,
            "r_multiple": (exit_price - entry) / risk,
            "net_rupees": gross - charges, "qty": qty,
            "held_days": exit_idx - j0 + 1, "exit_idx": exit_idx}


def _replay_symbol(symbol, df, max_hold, notional):
    """Every completed trade for one stock. Top-level so worker processes can run it."""
    rows = []
    if len(df) < signals.TREND_MA + max_hold + 5:
        return symbol, rows
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
                rows.append({"symbol": symbol, "signal": sig["signal"],
                             "notional": notional, **trade})
    return symbol, rows


def collect_trades(frames, max_hold=MAX_HOLD, notional=10000.0, workers=1, progress=False):
    """
    Run every detector over every symbol's history. frames: {symbol: indicator df}.

    One trade at a time per stock per rule: while a trade is open, the same rule
    firing again on that stock is ignored. Otherwise a setup that fires three days
    running counts as three trades, which inflates the sample size and makes a
    weak rule look statistically solid. Trades still open when the data ends are
    dropped, because they have no result yet.

    workers > 1 splits the stocks across CPU cores. The result is identical to
    workers=1; only the speed differs.
    """
    total, done = len(frames), 0
    by_symbol = {}

    def tick():
        if progress and (done % 50 == 0 or done == total):
            print(f"  replayed {done}/{total} stocks", flush=True)

    if workers <= 1:
        for symbol, df in frames.items():
            by_symbol[symbol] = _replay_symbol(symbol, df, max_hold, notional)[1]
            done += 1
            tick()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_replay_symbol, sym, df, max_hold, notional)
                       for sym, df in frames.items()]
            for fut in as_completed(futures):
                symbol, rows = fut.result()
                by_symbol[symbol] = rows
                done += 1
                tick()

    rows = [r for symbol in frames for r in by_symbol.get(symbol, [])]   # stable order
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


def naive_t(values):
    """Confidence that the mean is above zero, if every trade were independent."""
    x = np.asarray(values, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def clustered_t(values, clusters):
    """
    Confidence that the mean is above zero, treating each trading day as ONE
    piece of evidence. Trades entered on the same day rise and fall together
    with the market, so counting them separately overstates certainty.
    Standard cluster-robust estimate: se = sqrt(sum over days of (sum of residuals)^2) / n.
    """
    x = np.asarray(values, float)
    n = len(x)
    if n < 2:
        return float("nan")
    resid = pd.Series(x - x.mean())
    day_sums = resid.groupby(np.asarray(clusters)).sum()
    se = float(np.sqrt((day_sums ** 2).sum()) / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def price_at_size(trades, capital, risk_pct):
    """
    Re-price trades at the size you would actually trade: risk risk_pct of
    capital, never spend more than capital. Returns (qty, net rupees); qty 0
    means you could not afford even one share at that risk.
    """
    risk_cash = capital * risk_pct
    qty_risk = np.floor(risk_cash / trades["risk_per_share"])
    qty_cash = np.floor(capital / (trades["entry"] * 1.002))
    qty = np.maximum(0, np.minimum(qty_risk, qty_cash)).astype(int)
    buy, sell = trades["entry"] * qty, trades["exit"] * qty
    net = (trades["exit"] - trades["entry"]) * qty - costs.round_trip_cost(buy, sell)
    return qty, net.where(qty > 0)


def stress_test(trades, split=SPLIT_DATE, capital=2000.0, risk_pct=0.01):
    """Stability and significance checks for every rule that made money overall."""
    results = {}
    if trades.empty:
        return results
    split_ts = pd.Timestamp(split)
    for name, grp in trades.groupby("signal"):
        if grp["net_rupees"].mean() <= 0 or len(grp) < MIN_SAMPLE:
            continue                     # losers need no stress test; tiny samples can't have one
        early = grp[grp["entry_date"] < split_ts]["net_rupees"]
        late = grp[grp["entry_date"] >= split_ts]["net_rupees"]
        by_year = grp.groupby(grp["entry_date"].dt.year)["net_rupees"].agg(["mean", "size"])
        counted = by_year[by_year["size"] >= MIN_YEAR_TRADES]
        positive_years = int((counted["mean"] > 0).sum())
        t_clu = clustered_t(grp["net_rupees"], grp["entry_date"].dt.date)
        qty, net_small = price_at_size(grp, capital, risk_pct)
        affordable = net_small.dropna()

        reasons = []
        if len(early) < MIN_SAMPLE or len(late) < MIN_SAMPLE:
            reasons.append("too few trades in one of the two periods to compare")
        else:
            if early.mean() <= 0:
                reasons.append(f"lost money before {split}")
            if late.mean() <= 0:
                reasons.append(f"lost money from {split} on")
        if not (t_clu >= ROBUST_T):
            reasons.append(f"confidence {t_clu:.1f} is below {ROBUST_T:.0f} — could be luck")
        if len(counted) == 0:
            reasons.append(f"no single year had {MIN_YEAR_TRADES}+ trades to judge by")
        elif positive_years / len(counted) < ROBUST_YEAR_SHARE:
            reasons.append(f"profitable in only {positive_years} of {len(counted)} years")

        results[name] = {
            "early": {"samples": int(len(early)),
                      "per_trade": float(early.mean()) if len(early) else None},
            "late": {"samples": int(len(late)),
                     "per_trade": float(late.mean()) if len(late) else None},
            "by_year": {int(y): {"per_trade": float(r["mean"]), "samples": int(r["size"])}
                        for y, r in by_year.iterrows()},
            "positive_years": positive_years,
            "years_counted": int(len(counted)),
            "t_naive": naive_t(grp["net_rupees"]),
            "t_clustered": t_clu,
            "your_size": {"capital": capital, "risk_pct": risk_pct,
                          "affordable_share": float((qty > 0).mean()),
                          "samples": int(len(affordable)),
                          "per_trade": float(affordable.mean()) if len(affordable) else None},
            "notional": float(grp["notional"].iloc[0]) if "notional" in grp else 10000.0,
            "robust": not reasons,
            "reasons": reasons,
        }
    return results


def notional_label(r):
    return f"{r.get('notional', 10000):,.0f}"


def _rupees(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"Rs {v:+,.0f}"


def print_stress(results, split):
    if not results:
        print(f"STRESS TEST: no rule made money overall on {MIN_SAMPLE}+ trades, so there is")
        print("nothing to stress-test.\n")
        return
    print("STRESS TEST — only the rules that made money overall")
    print("Is the edge stable over time, and bigger than luck? A rule must pass all three:")
    print(f"profitable both before and after {split}, confidence at least {ROBUST_T:.0f},")
    print(f"and profitable in at least {ROBUST_YEAR_SHARE:.0%} of years.\n")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["t_clustered"]):
        ys = r["your_size"]
        small_loses = ys["per_trade"] is None or ys["per_trade"] <= 0
        if r["robust"] and small_loses:
            verdict = (f"HOLDS UP at Rs {notional_label(r)} per trade — but LOSES MONEY at "
                       f"your Rs {ys['capital']:,.0f}")
        else:
            verdict = "HOLDS UP" if r["robust"] else "FRAGILE"
        print(f"{name}: {verdict}")
        print(f"   before {split}: {_rupees(r['early']['per_trade'])} per trade "
              f"({r['early']['samples']:,} trades)   from {split}: "
              f"{_rupees(r['late']['per_trade'])} ({r['late']['samples']:,} trades)")
        years = "  ".join(f"{y} {v['per_trade']:+.0f}" for y, v in sorted(r["by_year"].items()))
        print(f"   by year: {years}")
        years_txt = (f"profitable in {r['positive_years']} of {r['years_counted']} years"
                     if r["years_counted"] else f"no year had {MIN_YEAR_TRADES}+ trades")
        print(f"   {years_txt}   "
              f"confidence {r['t_clustered']:.1f} (would look like {r['t_naive']:.1f} "
              f"if every trade were independent)")
        tag = ("LOSES MONEY" if ys["per_trade"] is not None and ys["per_trade"] <= 0
               else "makes money" if ys["per_trade"] is not None else "never affordable")
        print(f"   at your size (Rs {ys['capital']:,.0f}, {ys['risk_pct']:.0%} risk): "
              f"{ys['affordable_share']:.0%} of trades affordable, "
              f"{_rupees(ys['per_trade'])} per trade after charges — {tag}")
        for reason in r["reasons"]:
            print(f"   why fragile: {reason}")
        print()
    print("This split is a stability check, not a clean test: these rules were picked")
    print("after looking at all the years. The only clean test is trades the rules have")
    print("never seen — paper trade them from today and compare.\n")


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


def load_trades(path=TRADES_FILE):
    trades = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    notional = float(trades["notional"].iloc[0]) if "notional" in trades and len(trades) else None
    return trades, notional


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
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="CPU cores to use for the replay (default: all but one)")
    p.add_argument("--from-trades", action="store_true",
                   help="skip the slow replay and re-analyse the trades saved by the last run")
    p.add_argument("--split", default=SPLIT_DATE, help="stress test: compare before/after this date")
    p.add_argument("--capital", type=float, default=2000, help="stress test: your trading capital")
    p.add_argument("--risk-pct", type=float, default=0.01, help="stress test: risk per trade")
    args = p.parse_args()

    if args.from_trades:
        if not TRADES_FILE.exists():
            raise SystemExit(f"No {TRADES_FILE.name} yet. Run 'python evaluate.py' once without "
                             f"--from-trades first.")
        trades, saved_notional = load_trades()
        notional = saved_notional or args.notional
        print(f"Re-analysing {len(trades):,} saved trades from {TRADES_FILE.name} (no replay).")
    else:
        import data
        import universe
        min_turnover = universe.MIN_TURNOVER_CR if args.min_turnover is None else args.min_turnover
        frames = data.load_frames(symbols=args.symbols.split(",") if args.symbols else None,
                                 years=args.years, demo=args.demo, min_turnover_cr=min_turnover)
        if not frames:
            raise SystemExit("No price data available. Try --demo, or run scan.py --update first.")
        workers = min(args.workers, len(frames))
        print(f"Replaying {len(frames)} stocks on {workers} CPU core(s)...", flush=True)
        trades = collect_trades(frames, args.max_hold, args.notional,
                                workers=workers, progress=True)
        notional = args.notional
        if not trades.empty:
            trades.to_csv(TRADES_FILE, index=False)
            # reload so dtypes match a --from-trades run exactly
            trades, _ = load_trades()

    stats = summarize(trades)
    print_stats(stats, notional)
    stress = stress_test(trades, args.split, args.capital, args.risk_pct)
    print_stress(stress, args.split)
    for name, st in stats.items():
        st["robust"] = bool(stress.get(name, {}).get("robust", False))
        if name in stress:
            st["stress"] = stress[name]
    STATS_FILE.write_text(json.dumps({"notional": notional, "signals": stats}, indent=2))
    print(f"Saved to {STATS_FILE.name} — scan.py reads it to score today's picks.")
    if not args.from_trades and not trades.empty:
        print(f"Saved {len(trades):,} trades to {TRADES_FILE.name}. Next time, "
              f"'python evaluate.py --from-trades' re-analyses them in seconds.")
    if args.trades_csv and not trades.empty:
        trades.to_csv(args.trades_csv, index=False)
        print(f"Wrote {len(trades)} trades to {args.trades_csv}")


if __name__ == "__main__":
    main()

"""
Today's swing-trade candidates, with each rule's real historical record.

It places NO orders and never touches your Kite login. It tells you what your
written rules say and how those rules have performed in the past. The decision,
and the consequences, stay yours.

By default it only shows setups from rules that were measured on enough history
AND made money after charges. Rules that lost money historically are hidden,
because showing them next to the winners invites you to take them anyway.

First time:
  pip install pandas yfinance requests
  python scan.py --update --refresh-universe     # download prices (slow)
  python evaluate.py                             # measure the rules on history
  python scan.py                                 # today's top 5

Every day, after the market closes:
  python scan.py --update && python scan.py

One stock:
  python scan.py --symbol RELIANCE
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import costs
import data
import signals
import universe

STATS_FILE = Path(__file__).with_name("stats.json")
STALE_DAYS = 5          # a stock whose last bar is this many days old is not "today"
CHARGE_WARN = 0.25      # warn when charges eat this share of the rupees you are risking


def load_stats():
    """(per-signal stats, rupees per trade they were measured at)."""
    if not STATS_FILE.exists():
        return {}, None
    try:
        raw = json.loads(STATS_FILE.read_text())
        return raw.get("signals", {}), raw.get("notional")
    except (json.JSONDecodeError, OSError):
        return {}, None


def proven(sig_name, stats):
    """
    Measured on enough history, made money after charges, and — when the stress
    test has been run — held up across time and beyond luck.
    """
    st = stats.get(sig_name)
    if not (st and st.get("reliable") and st.get("expectancy_rupees", 0) > 0):
        return False
    return st.get("robust", True)          # older stats.json files have no stress test


def position_size(entry, stop, capital, risk_pct):
    """How many shares risks exactly risk_pct of capital, capped by what you can afford."""
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    by_risk = int((capital * risk_pct) // risk_per_share)
    by_cash = int(capital // (entry * 1.002))          # leave room for charges
    return max(0, min(by_risk, by_cash))


def score(sigs, stats):
    """Rank by the best historical expectancy among the rules that fired."""
    best = max(stats.get(s["signal"], {}).get("expectancy_r", -1.0) for s in sigs)
    return best + 0.1 * (len(sigs) - 1)                 # small bonus when rules agree


def fresh_frames(frames):
    """Drop stocks whose data stops before the latest trading day (suspended, delisted)."""
    if not frames:
        return {}, 0, None
    latest = max(df.index[-1] for df in frames.values())
    kept = {s: df for s, df in frames.items()
            if (latest - df.index[-1]).days <= STALE_DAYS}
    return kept, len(frames) - len(kept), latest


def find_setups(frames, stats, include_unproven=False):
    """
    Today's setups, best first, as (score, symbol, df, signals). Unless
    include_unproven, only rules with a proven record are kept (when stats exist).
    Returns (setups, number of setups hidden by that filter).
    """
    filter_proven = bool(stats) and not include_unproven
    found, hidden = [], 0
    for sym, df in frames.items():
        sigs = signals.detect_all(df, len(df) - 1)
        if filter_proven:
            kept = [s for s in sigs if proven(s["signal"], stats)]
            hidden += len(sigs) - len(kept)
            sigs = kept
        if sigs:
            found.append((score(sigs, stats), sym, df, sigs))
    found.sort(key=lambda t: -t[0])
    return found, hidden


def describe(symbol, df, sigs, stats, notional, capital, risk_pct, min_turnover, rank=None):
    head = f"{rank}. " if rank else ""
    names = " + ".join(s["signal"] for s in sigs)
    primary = max(sigs, key=lambda s: stats.get(s["signal"], {}).get("expectancy_r", -99))
    entry, stop, target = primary["trigger"], primary["stop"], primary["target"]
    risk = entry - stop
    qty = position_size(entry, stop, capital, risk_pct)
    lines = [f"{head}{symbol:12} {names}",
             f"   buy above {entry:>10.2f}   stop {stop:>10.2f}   target {target:>10.2f}",
             f"   risk/share Rs {risk:.2f}"]
    if qty > 0:
        charges = costs.round_trip_cost(entry * qty, target * qty)
        at_risk = risk * qty
        lines.append(f"   qty at {risk_pct:.0%} risk of Rs {capital:,.0f}: {qty}"
                     f"   (risking Rs {at_risk:.0f}, charges about Rs {charges:.0f})")
        if charges >= CHARGE_WARN * at_risk:
            lines.append(f"   WARNING: charges are {charges / at_risk:.0%} of what you would risk. "
                         f"Too small to be worth trading — paper trade it.")
    else:
        lines.append(f"   qty: 0 — Rs {capital:,.0f} cannot size this trade. PAPER TRADE ONLY.")

    per = f" per Rs {notional:,.0f} trade" if notional else ""
    for s in sigs:
        st = stats.get(s["signal"])
        if not st:
            lines.append(f"   {s['signal']}: never measured — run evaluate.py first")
        elif not st["reliable"]:
            lines.append(f"   {s['signal']}: only {st['samples']} past cases — not enough to trust")
        else:
            verdict = "profitable historically" if st["expectancy_rupees"] > 0 else "LOST MONEY historically"
            lines.append(f"   {s['signal']}: {st['samples']} past cases, {st['win_rate']:.0%} won, "
                         f"avg win Rs {st['avg_win_rupees']:,.0f} / avg loss Rs "
                         f"{abs(st['avg_loss_rupees']):,.0f}{per}, "
                         f"{st['expectancy_r']:+.2f}R before charges — {verdict} after charges")
    t = universe.turnover_crores(df)
    note = "fine" if t >= min_turnover else "THIN — you may not be able to sell when you want"
    lines.append(f"   traded value Rs {t:,.1f} cr/day — {note}")
    return "\n".join(lines)


def clean_symbol(raw):
    """'reliance', ' RELIANCE.NS ' and 'RELIANCE' all mean RELIANCE."""
    sym = raw.strip().upper()
    return sym[:-3] if sym.endswith(".NS") else sym


def analyse_one(args, stats, notional):
    frames = data.load_frames([args.symbol], years=args.years, demo=args.demo, quiet=True)
    if args.symbol not in frames:
        hint = " Demo stocks are DEMO1 to DEMO12." if args.demo else ""
        raise SystemExit(f"No cached data for {args.symbol}.{hint} Run:  python scan.py "
                         f"--symbol {args.symbol} --update")
    sym, df = args.symbol, frames[args.symbol]
    last = len(df) - 1
    age = (pd.Timestamp.today().normalize() - df.index[last]).days
    print(f"\n{sym} on {df.index[last].date()}   close {df['Close'].iat[last]:.2f}")
    if age > STALE_DAYS and not args.demo:
        print(f"   WARNING: this data is {age} days old. Run with --update before trusting it.")
    trend = "above" if df["Close"].iat[last] > df["trend_ma"].iat[last] else "below"
    print(f"   {trend} its {signals.TREND_MA}-day average"
          f" | RSI {df['rsi'].iat[last]:.0f}"
          f" | traded value Rs {universe.turnover_crores(df):,.1f} cr/day\n")
    sigs = signals.detect_all(df, last)
    if sigs:
        print(describe(sym, df, sigs, stats, notional, args.capital, args.risk_pct,
                       args.min_turnover))
        if stats and not any(proven(s["signal"], stats) for s in sigs):
            print("\n   None of these rules has a proven profitable record. The scan would not")
            print("   recommend this stock today.")
    else:
        print("No setup here today. That is a real answer, not a failure —")
        print("the rules say wait. Forcing a trade is how accounts bleed.")
    print()


def main():
    p = argparse.ArgumentParser(description="Find today's swing setups. Suggests only, never trades.")
    p.add_argument("--symbol", help="analyse one stock instead of scanning, e.g. RELIANCE")
    p.add_argument("--symbols", help="scan only these comma separated symbols")
    p.add_argument("--capital", type=float, default=2000, help="your trading capital in Rs")
    p.add_argument("--risk-pct", type=float, default=0.01, help="fraction of capital risked per trade")
    p.add_argument("--min-turnover", type=float, default=universe.MIN_TURNOVER_CR,
                   help="skip stocks trading under this many Rs crore per day")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--include-unproven", action="store_true",
                   help="also show setups from rules that lost money or lack enough history")
    p.add_argument("--update", action="store_true", help="download fresh prices into the cache first")
    p.add_argument("--refresh-universe", action="store_true", help="re-fetch the NSE symbol list")
    p.add_argument("--demo", action="store_true", help="synthetic prices, no internet needed")
    args = p.parse_args()
    if args.symbol:
        args.symbol = clean_symbol(args.symbol)
    if args.symbols:
        args.symbols = ",".join(clean_symbol(s) for s in args.symbols.split(",") if s.strip())

    if args.update or args.refresh_universe:
        syms = ([args.symbol] if args.symbol else
                args.symbols.split(",") if args.symbols else
                universe.load_symbols(refresh=args.refresh_universe))
        if not (args.symbol or args.symbols) and data.BENCHMARK not in syms:
            syms = list(syms) + [data.BENCHMARK]     # the research lab compares against it
        print(f"Updating prices for {len(syms)} symbols. This takes a while for the full list.")
        ok, failed = data.update_cache(syms, years=args.years)
        print(f"Done: {ok} cached, {failed} unavailable.\n")

    stats, notional = load_stats()
    if not stats:
        print("WARNING: stats.json missing, so no historical odds can be shown and")
        print("nothing can be filtered for a proven record. Run 'python evaluate.py'")
        print("first — without it you are trading blind.\n")

    if args.symbol:
        analyse_one(args, stats, notional)
        return

    symbols = args.symbols.split(",") if args.symbols else None
    frames = data.load_frames(symbols, years=args.years, demo=args.demo,
                              min_turnover_cr=args.min_turnover)
    frames, stale, latest = fresh_frames(frames)
    if not frames:
        raise SystemExit("No stocks passed the filters. Run with --update, or lower --min-turnover.")

    found, hidden = find_setups(frames, stats, include_unproven=args.include_unproven)

    if not stats:
        print(f"\nUNMEASURED SETUPS FOR {latest.date()} — NOT RECOMMENDATIONS")
        print("None of these rules has been checked against history yet. The order below")
        print("only counts how many rules agree; it says nothing about quality.")
    print(f"\nSETUPS FOR {latest.date()} — {len(frames)} stocks checked, {len(found)} with a setup")
    if stale:
        print(f"({stale} stocks skipped: their data stops before {latest.date()}, "
              f"likely suspended or delisted)")
    if hidden:
        print(f"({hidden} setups hidden: their rules lost money, lack enough history, or "
              f"failed the stress test. --include-unproven shows them)")
    print()
    if not found:
        print("Nothing qualifies today. Do nothing. This is the correct output on most days.\n")
        return
    for rank, (_, sym, df, sigs) in enumerate(found[:args.top], 1):
        print(describe(sym, df, sigs, stats, notional, args.capital, args.risk_pct,
                       args.min_turnover, rank))
        print()
    if not stats:
        print("Do not trade any of these. Run 'python evaluate.py' first, then scan again.\n")
        return
    print("Place these yourself in Kite as DELIVERY (Longterm) orders, and set a")
    print("GTT for the stop and target the moment you buy, so the exit is automatic.")
    print("These are historical odds, not predictions. Some of these will lose.\n")


if __name__ == "__main__":
    main()

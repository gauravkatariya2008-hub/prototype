"""
Moving-average crossover backtester for NSE stocks — a learning tool.

This script does NOT place orders and does NOT connect to Kite / Zerodha.
It replays past prices to show how a simple rule-based strategy *would have*
done, including approximate Indian delivery trading costs, and compares it
with simply buying and holding the same stock.

The rule (long only, no leverage, no short selling):
  - BUY  when the fast moving average crosses ABOVE the slow moving average
  - SELL when the fast moving average crosses BELOW the slow moving average
The signal is calculated on a day's closing price and executed at the NEXT
day's opening price, because you cannot trade at a close you haven't seen yet.

Usage:
  pip install pandas yfinance
  python backtest.py                              # RELIANCE, last 5 years
  python backtest.py --symbol TCS.NS --years 10
  python backtest.py --symbol INFY.NS --fast 10 --slow 30
  python backtest.py --csv my_prices.csv          # your own data (Date,Open,High,Low,Close)
  python backtest.py --demo                       # made-up prices, no internet needed
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd

from costs import STT, STAMP_DUTY_BUY, trade_cost


def load_prices(args):
    if args.demo:
        return demo_prices(args.years)
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["Date"], index_col="Date")
        return df[["Open", "Close"]].dropna().sort_index()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. Run:  pip install yfinance")

    df = yf.download(args.symbol, period=f"{args.years}y", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        sys.exit(f"No data for {args.symbol}. NSE symbols end in .NS (e.g. TCS.NS, HDFCBANK.NS).")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "Close"]].dropna()


def demo_prices(years):
    """Random-walk prices, so the script can be tried without internet."""
    rng = np.random.default_rng(42)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=years * 250)
    close = 1000 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(days))))
    open_ = close * (1 + rng.normal(0, 0.004, len(days)))
    return pd.DataFrame({"Open": open_, "Close": close}, index=days)


def run_backtest(prices, fast, slow, capital):
    df = prices.copy()
    df["fast_ma"] = df["Close"].rolling(fast).mean()
    df["slow_ma"] = df["Close"].rolling(slow).mean()
    # 1 = fast above slow (want to be holding), 0 = otherwise
    df["want_position"] = (df["fast_ma"] > df["slow_ma"]).astype(int)

    cash, shares = capital, 0
    total_costs = 0.0
    trades, equity = [], []
    entry = None

    dates = df.index
    for i in range(len(df)):
        # Act today at the open on YESTERDAY's signal (no peeking at today's close)
        if i > 0 and not math.isnan(df["slow_ma"].iloc[i - 1]):
            want = df["want_position"].iloc[i - 1]
            price = df["Open"].iloc[i]

            if want == 1 and shares == 0:
                qty = int(cash // (price * (1 + STT + STAMP_DUTY_BUY + 0.0001)))
                if qty > 0:
                    cost = trade_cost(qty * price, "buy")
                    cash -= qty * price + cost
                    total_costs += cost
                    shares = qty
                    entry = (dates[i], price, qty, cost)

            elif want == 0 and shares > 0:
                cost = trade_cost(shares * price, "sell")
                cash += shares * price - cost
                total_costs += cost
                buy_date, buy_price, qty, buy_cost = entry
                pnl = (price - buy_price) * qty - buy_cost - cost
                trades.append({"buy_date": buy_date.date(), "buy_price": buy_price,
                               "sell_date": dates[i].date(), "sell_price": price,
                               "qty": qty, "pnl": pnl})
                shares, entry = 0, None

        equity.append(cash + shares * df["Close"].iloc[i])

    df["equity"] = equity
    return df, trades, total_costs, shares


def summarize(equity, capital):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    final = equity.iloc[-1]
    total_return = final / capital - 1
    cagr = (final / capital) ** (1 / years) - 1 if years > 0 and final > 0 else float("nan")
    drawdown = (equity / equity.cummax() - 1).min()
    return final, total_return, cagr, drawdown


def main():
    p = argparse.ArgumentParser(description="Backtest a moving-average crossover on an NSE stock.")
    p.add_argument("--symbol", default="RELIANCE.NS", help="Yahoo Finance symbol, NSE stocks end in .NS")
    p.add_argument("--years", type=int, default=5, help="how many years of history to test")
    p.add_argument("--fast", type=int, default=20, help="fast moving average length (days)")
    p.add_argument("--slow", type=int, default=50, help="slow moving average length (days)")
    p.add_argument("--capital", type=float, default=100000, help="starting money in Rs")
    p.add_argument("--csv", help="use prices from a CSV file instead of downloading")
    p.add_argument("--demo", action="store_true", help="use made-up prices (no internet needed)")
    args = p.parse_args()

    if args.fast >= args.slow:
        sys.exit("--fast must be smaller than --slow")

    prices = load_prices(args)
    if len(prices) <= args.slow + 1:
        sys.exit(f"Only {len(prices)} days of data — need more than {args.slow + 1}.")

    df, trades, costs, open_shares = run_backtest(prices, args.fast, args.slow, args.capital)

    # Buy and hold: buy on the same day the strategy could first trade, never sell
    start = df["slow_ma"].first_valid_index()
    start_i = df.index.get_loc(start) + 1
    bh_price = df["Open"].iloc[start_i]
    bh_qty = int(args.capital // (bh_price * (1 + STT + STAMP_DUTY_BUY + 0.0001)))
    bh_cash = args.capital - bh_qty * bh_price - trade_cost(bh_qty * bh_price, "buy")
    bh_equity = bh_cash + bh_qty * df["Close"].iloc[start_i:]

    strat_equity = df["equity"].iloc[start_i:]
    s_final, s_ret, s_cagr, s_dd = summarize(strat_equity, args.capital)
    b_final, b_ret, b_cagr, b_dd = summarize(bh_equity, args.capital)

    name = "DEMO (random prices)" if args.demo else (args.csv or args.symbol)
    print(f"\n{name}: {strat_equity.index[0].date()} to {strat_equity.index[-1].date()}")
    print(f"Rule: buy when {args.fast}-day average crosses above {args.slow}-day average, sell when it crosses below")
    print(f"Starting money: Rs {args.capital:,.0f}\n")

    print(f"{'':22}{'Strategy':>14}{'Buy & hold':>14}")
    print(f"{'Final value (Rs)':22}{s_final:>14,.0f}{b_final:>14,.0f}")
    print(f"{'Total return':22}{s_ret:>14.1%}{b_ret:>14.1%}")
    print(f"{'Yearly return (CAGR)':22}{s_cagr:>14.1%}{b_cagr:>14.1%}")
    print(f"{'Worst fall from peak':22}{s_dd:>14.1%}{b_dd:>14.1%}")

    wins = [t for t in trades if t["pnl"] > 0]
    print(f"\nCompleted trades: {len(trades)}   winners: {len(wins)}   losers: {len(trades) - len(wins)}")
    print(f"Taxes and charges paid by the strategy: Rs {costs:,.0f}")
    if open_shares:
        print(f"Still holding {open_shares} shares at the end (counted at last closing price).")

    if trades:
        print("\nLast 10 trades:")
        for t in trades[-10:]:
            print(f"  bought {t['qty']:>4} on {t['buy_date']} @ {t['buy_price']:>9.2f}  "
                  f"sold on {t['sell_date']} @ {t['sell_price']:>9.2f}  P&L Rs {t['pnl']:>10,.0f}")

    print("\nReminder: past results do not predict future results, and this ignores")
    print("income tax on gains (STCG/LTCG). Backtest many stocks and periods before")
    print("believing any rule works.\n")


if __name__ == "__main__":
    main()

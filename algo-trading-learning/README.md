# Swing-Trade Scanner for NSE Stocks

A learning tool. It **never places orders** and never touches your Zerodha login.

It checks written-down trading rules against NSE stocks every evening. For each setup it finds, it shows you **how that exact rule performed on years of past data, after Zerodha charges**. It doesn't predict the future and doesn't promise profit. It tells you the measured odds, so you can decide.

## What's in here

| File | What it does |
|---|---|
| `scan.py` | Today's top setups, or analyses one stock you name |
| `evaluate.py` | Measures every rule on past data: win rate, average win and loss, profit after charges |
| `signals.py` | The rules as code. Every threshold is a named number at the top |
| `patterns.md` | The same rules in plain English. Check them against your Varsity notes |
| `costs.py` | Zerodha delivery charges (STT, stamp duty, exchange fees, DP charge) |
| `data.py`, `universe.py` | Price downloads, the local price cache and the liquidity filter |
| `backtest.py` | The simpler moving-average backtester from earlier |
| `tests.py` | 65 tests. Run them after changing anything |

## Setup (once)

You need Python 3.

```bash
pip install pandas yfinance requests
python scan.py --update --refresh-universe   # downloads all NSE stocks, ~10-20 minutes
python evaluate.py                           # measures every rule on the history
```

`evaluate.py` uses all but one of your CPU cores and prints progress every 50 stocks. It saves every simulated trade to `trades.csv`. After that, `python evaluate.py --from-trades` re-analyses them in seconds without replaying anything.

Try it without internet first with `python evaluate.py --demo` and `python scan.py --demo`. These use made-up prices.

## Every day, in the evening after the market closes

```bash
python scan.py --update     # refresh prices (slow for the full list)
python scan.py              # today's top 5
```

To check one stock:

```bash
python scan.py --symbol RELIANCE
```

Useful options:
- `--capital 2000`: your trading money, used to size positions.
- `--risk-pct 0.01`: risk 1% of capital per trade.
- `--min-turnover 5`: skip stocks trading under ₹5 crore a day.
- `--top 5`: how many picks to show.

## Reading the output

- **buy above / stop / target:** place a buy order only if the price goes above the trigger. The moment it fills, set a **GTT** in Kite (an order that waits until your price is hit) for the stop and target, so the exit happens automatically.
- **past cases, % won, avg win / avg loss:** how this exact rule did historically.
- **R before charges:** the move in multiples of your risk, before charges.
- **profitable / LOST MONEY:** judged after charges. This decides whether the scanner recommends it.
- **qty: 0, PAPER TRADE ONLY:** your capital is too small to trade this safely. Write it down and track it without real money.
- **Nothing qualifies today:** the correct answer on most days. It's not a malfunction.

## Reading the stress test

After the main table, `evaluate.py` stress-tests every rule that made money. A rule **holds up** only if it passes all three checks:

- **Profitable both before and after the split date** (`--split`, default 2023-01-01). An edge that only showed up in some years is probably luck.
- **Confidence of at least 2.** Trades entered on the same day move together with the market, so each trading day counts as one piece of evidence. The table also shows the inflated figure you'd get if every trade counted separately.
- **Profitable in at least 60% of years.**

It also re-prices every trade at your own size (`--capital`, `--risk-pct`). A rule can hold up at ₹10,000 per trade and still lose money at ₹2,000, because of the flat ₹15 DP charge. The verdict line says so when that happens.

The split is a stability check, not a clean test, because these rules were chosen after looking at all the years. The only clean test is trades the rules have never seen, which means paper trading from today.

## Which setups the scanner shows

By default the scanner **only shows setups from rules that were tested on at least 100 past cases, made money after charges, and passed the stress test.** Losing rules are hidden. `--include-unproven` shows them, clearly labelled.

## What it can't do

- **Predict anything.** Every number is historical. Markets change, and a rule that worked for 8 years can stop working.
- **Undo bias in the data.** Past prices only exist for companies still listed today. Delisted failures are missing, so results look better than reality.
- **Promise that a profitable-historically rule will make you money.** It means the odds leaned your way in the past, nothing more.
- **Trade intraday.** That needs live data (Kite Connect, about ₹2,000/month). This tool is for trades held for days.
- **See real data in the environment where it was built.** Yahoo Finance was blocked there, so it was tested on synthetic and simulated data. The first run on real prices happens on your machine.

## Checks built in (so the numbers can be trusted)

- **No look-ahead:** signals use only data up to the day they fire. This is tested by deleting future days and confirming nothing changes.
- **Next-day entry:** you enter at the next day's price, never the same day's close.
- **Pessimistic when unsure:** if the stop and the target were both hit on the same day, it counts as a loss.
- **No double-counting:** one trade at a time per stock per rule. Trades still open when the data ends are excluded.
- **Liquidity filter and stale-data filter:** suspended or delisted stocks can't appear as today's picks.
- **Charges deducted** on both the buy and the sell.

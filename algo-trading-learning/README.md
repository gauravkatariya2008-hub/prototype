# Algo Trading — Beginner Learning Kit

This is a **learning tool, not a money-making bot.** It never places orders and never connects to your Zerodha account.

## What algo trading is

Algo (algorithmic) trading means writing your buy and sell rules as code, so a computer applies them instead of you deciding by feel. For example:

> "Buy when the 20-day average price rises above the 50-day average. Sell when it drops below."

A program checks that rule and acts on it. The code only follows your rules faster and without emotion. **If the rules lose money, the code loses money faster.**

The serious work in algo trading is **backtesting**: running your rule on years of past prices to see what would have happened, before you put in real money. That is what this script does.

## Running it

You need Python 3 installed.

```bash
pip install pandas yfinance
python backtest.py --demo                       # made-up prices, to check it runs
python backtest.py                              # Reliance, last 5 years
python backtest.py --symbol TCS.NS --years 10
python backtest.py --symbol HDFCBANK.NS --fast 10 --slow 30
```

NSE symbols end in `.NS` (for example `INFY.NS` or `SBIN.NS`). Price data comes from Yahoo Finance.

## Reading the output

| Line | Meaning |
|---|---|
| Strategy vs Buy & hold | Did the rule beat simply buying once and holding? Often it doesn't. |
| Yearly return (CAGR) | Average growth per year. Compare it with a NIFTY index fund or an FD. |
| Worst fall from peak | The biggest drop you would have had to sit through. Ask yourself whether you could stomach it. |
| Taxes and charges | STT, stamp duty, exchange fees and DP charges. Frequent trading adds these up quickly. |

## How the script avoids lying to you

- It trades at the **next day's open**, not the same day's close. You can't act on a closing price before you've seen it.
- It deducts **approximate delivery charges** (check Zerodha's brokerage calculator for current rates).
- It only buys and sells shares. There's no leverage, no short selling and no F&O.

It does **not** include income tax on gains (STCG/LTCG), and it can't account for the future looking different from the past.

## Before you ever go live

1. Learn Kite itself first. Zerodha Varsity (zerodha.com/varsity) is free and is the best starting point in India.
2. Backtest across many stocks and time periods. One good result on one stock is usually luck.
3. Paper trade: write down what the rule says each day for a few months, without real money.
4. Stay away from F&O (futures and options) as a beginner. SEBI's own studies found that about 9 out of 10 individual F&O traders lost money.
5. If you ever automate real orders through the Kite Connect API, read SEBI's current rules for retail algo trading first. Brokers must now approve and tag algo orders.

# Project context for Claude Code

NSE Strategy Research Lab: a Streamlit dashboard plus command-line tools that test trading ideas on past NSE prices with Zerodha charges. See README.md for features and commands.

## Hard rules from the owner

- **No order placement or auto-trading.** The owner explicitly said "not yet". The app suggests only. There is no broker or Kite Connect code anywhere, and `tests.py` fails if any appears. Don't add it unless the owner asks for it directly.
- **Be honest. Never promise accurate or profitable predictions.** Stock picks may only come from a strategy that beat buying NIFTYBEES on the owner's own data (see `portfolio.judge`). If none passes, the answer is NIFTYBEES. Never invent picks.
- **News is a warning system, not a stock picker.** News-based picks can't be backtested.
- **Answer straight, without sugarcoating.** The owner asked for this. When asking the owner to make a choice, ask one question at a time in a poll.

## Owner's situation

- A beginner, on Windows (Command Prompt), with a Zerodha Kite account and a budget of about ₹2,000.
- At ₹2,000, the flat ₹15.34 DP charge per sell makes small stock positions lose money. `picks.plan_budget` falls back to NIFTYBEES below about ₹3,000 per stock.

## What real data has shown so far

These results come from 928 liquid NSE stocks over about 8 years.

- **All 7 candlestick rules failed** (`evaluate.py`).
  - pullback, breakout, piercing and hammer lost money after charges.
  - morning_star, rsi_bounce and engulfing made a little money overall but failed the stress test:
    - morning_star: +₹40 per trade before 2023, −₹2 from 2023, profitable in 3 of 9 years, clustered confidence 1.6.
    - rsi_bounce: profitable in 4 of 8 years, confidence 1.2.
    - engulfing: confidence 0.5.
  - Most of the profit was in the 2020–21 post-COVID rally.
- **The portfolio strategies in the dashboard have not been run on real data yet.** They are momentum, low volatility, index trend and mean reversion.
- **The news feeds have never been fetched live.** The build environment blocked them, so they were only tested against sample RSS.

## How this code is kept trustworthy

- Run `python tests.py` after any change. All tests must pass.
- For every new rule or condition, add a test, then mutation-check it: delete the condition and confirm a test fails. This has caught real bugs several times.
- Backtest conventions:
  - decide on the close, trade at the next day's open;
  - whole shares only;
  - charges from `costs.py`, with ETFs at their lower STT;
  - liquidity judged point-in-time (60 days before each decision);
  - NIFTYBEES is never held by stock strategies;
  - the race against the benchmark starts only once a strategy has enough history;
  - a strategy "beats NIFTYBEES" only if it leads before and after the split date, has monthly t ≥ 2, and wins 60%+ of full years.
- After changing `app.py`, run the dashboard and look at it in both light and dark mode before calling it done.

## Running things

```
pip install -r requirements.txt
python scan.py --update --refresh-universe            # prices for all NSE stocks + NIFTYBEES + industries
python -c "import universe; universe.refresh_industries()"   # industries only
python evaluate.py                                    # measure candlestick rules
python evaluate.py --from-trades                      # re-analyse saved trades in seconds
streamlit run app.py                                  # the dashboard
```

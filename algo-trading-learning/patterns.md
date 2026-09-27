# The rules, in plain English

Check these against your Varsity notes. Where they differ, tell me, or change the numbered constant at the top of `signals.py` yourself. Then re-run `python evaluate.py` to measure the new version. Never change a rule without re-measuring it.

**How every rule is used:**
- It's checked on a day's **closing** candle.
- You act the **next** day, and only buy if the price trades **above the trigger** (usually that candle's high).
- **Stop loss:** where the rule says it was wrong.
- **Target:** 2 × the risk above the trigger (`REWARD_RISK`).
- Every rule also needs the stock to **close above its 50-day average** (`TREND_MA`). All of these are buy-the-dip-in-an-uptrend rules. None of them buy a stock that is falling.
- A setup is thrown out if its stop is closer than 0.5% (`MIN_RISK_PCT`) or further than 12% (`MAX_RISK_PCT`) from the trigger.

## Trend pullback
- Close above the 50-day average.
- Within the last 3 days (`PULLBACK_LOOKBACK`), a low came within 1% of the 20-day average (`PULLBACK_TOUCH`).
- Today is a green candle that closes above yesterday's close.
- **Trigger:** today's high. **Stop:** today's low.

## Bullish engulfing
*Varsity: Technical Analysis module, multiple candlestick patterns.*
- Yesterday red, today green.
- Today opens at or below yesterday's close and closes at or above yesterday's open, so it swallows yesterday's body.
- **Added by me, not Varsity:** today's body must be at least 1.3× yesterday's (`ENGULF_BODY_RATIO`). Without it, a candle barely larger than the previous one counts, which is noise.
- **Trigger:** today's high. **Stop:** the lower of the two lows.

## Hammer
*Varsity: single candlestick patterns.*
- Lower wick at least 2× the body (`HAMMER_WICK_RATIO`).
- Upper wick at most 0.5× the body (`HAMMER_UPPER_WICK`).
- **Changed from Varsity:** Varsity describes the hammer at the bottom of a *downtrend*. Here it must appear in an uptrend, near support. That makes it a "dip bought in an uptrend" signal, consistent with the other rules.
- Near support means the low is no more than 3% above the 20-day average (`HAMMER_NEAR_MA`).
- **Trigger:** high. **Stop:** low.

## Piercing pattern
- Yesterday red.
- Today opens below yesterday's close, then closes above the **midpoint** of yesterday's body but below yesterday's open.
- **Trigger:** today's high. **Stop:** the lower of the two lows.

## Morning star
- Day 1 is red.
- Day 2 is small, with a body no more than 30% of day 1's (`STAR_SMALL_BODY`).
- Day 3 is green and closes above the midpoint of day 1's body.
- **Relaxed from the strict textbook form:** no gap is required between the candles. Strict versions demand gaps, which large Indian stocks rarely make, so the pattern would almost never trigger. Check how your Varsity notes treat this, and tell me if they say otherwise.
- **Trigger:** day 3's high. **Stop:** the lowest low of the three days.

## Breakout on volume
- Today's close is the highest close of the last 20 days (`BREAKOUT_WINDOW`).
- Volume is at least 2× its 20-day average (`BREAKOUT_VOL_MULT`).
- **Trigger:** today's high.
- **Stop:** the higher (tighter) of the lowest low of the last 5 days, and trigger − 1.5 × ATR (`BREAKOUT_ATR_MULT`). ATR (average true range) is how much the stock typically moves in a day.

## RSI oversold bounce
- RSI(14) dipped below 35 at some point in the last 5 days (`RSI_OVERSOLD`, `RSI_LOOKBACK`).
- The stock is still above its 50-day average.
- RSI is rising today, on a green candle.
- **Trigger:** today's high. **Stop:** the lowest low of the last 5 days.

**Heads up, found while testing:** this rule fires **rarely**. RSI only drops below 35 when losses swamp recent gains, and a drop that big usually breaks below the 50-day average too, which disqualifies it. It fires in one narrow shape: a strong rally, a quiet sideways pause, then a sharp but shallow dip. In a search over 96 price shapes, only 10 triggered it. Expect few samples, which means weaker evidence.

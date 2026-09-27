"""
The trading rules, written as numbers instead of opinions.

Every rule here is evaluated on the CLOSE of a day and is meant to be acted on
at the NEXT day's open. Nothing in this file looks at a future bar — that bug
(look-ahead bias) is what makes worthless strategies look brilliant.

Each detector returns None (no setup) or a dict:
    {"signal", "trigger", "stop", "target"}
  trigger — buy only if price trades above this
  stop    — sell here if price falls to it (your maximum loss)
  target  — sell here if price rises to it

The exact rules in plain English are in patterns.md. Check that file against
your own Varsity notes and change the constants below if you disagree.
"""

import numpy as np
import pandas as pd

# --- tunable thresholds (change these, then re-run evaluate.py to re-measure) ---
TREND_MA = 50            # a stock must close above this average to be "in an uptrend"
PULLBACK_MA = 20         # the average price pulls back to
PULLBACK_LOOKBACK = 3    # how recently the pullback must have happened (bars)
PULLBACK_TOUCH = 1.01    # "touched" the average means within 1% of it
ENGULF_BODY_RATIO = 1.3  # today's body must be this much bigger than yesterday's
HAMMER_WICK_RATIO = 2.0  # lower wick at least 2x the body
HAMMER_UPPER_WICK = 0.5  # upper wick at most 0.5x the body
HAMMER_NEAR_MA = 1.03    # hammer's low at most 3% above the 20-day average (near support)
STAR_SMALL_BODY = 0.3    # the middle candle's body vs the first candle's body
BREAKOUT_WINDOW = 20     # breakout above the highest close of this many days
BREAKOUT_VOL_MULT = 2.0  # volume must be this multiple of its 20-day average
BREAKOUT_ATR_MULT = 1.5  # fallback stop distance in ATRs
RSI_PERIOD = 14
RSI_OVERSOLD = 35        # RSI must have dipped below this
RSI_LOOKBACK = 5         # ...within this many days
REWARD_RISK = 2.0        # target sits this many times the risk above entry
MIN_RISK_PCT = 0.005     # reject setups whose stop is under 0.5% away (too tight to be real)
MAX_RISK_PCT = 0.12      # reject setups risking more than 12% per share (too wide to size)


def rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0   # only gains -> maximum
    out[(avg_gain == 0) & (avg_loss > 0)] = 0.0     # only losses -> minimum
    return out


def atr(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df):
    """Attach every indicator the detectors need. Input needs Open/High/Low/Close/Volume."""
    out = df.copy()
    close = out["Close"]
    out["trend_ma"] = close.rolling(TREND_MA).mean()
    out["pull_ma"] = close.rolling(PULLBACK_MA).mean()
    out["rsi"] = rsi(close)
    out["atr"] = atr(out)
    out["vol_avg"] = out["Volume"].rolling(BREAKOUT_WINDOW).mean()
    out["close_high"] = close.rolling(BREAKOUT_WINDOW).max()
    # median rupee value traded per day — how easily you could actually get in and out
    out["turnover"] = (close * out["Volume"]).rolling(60).median()
    return out


def _body(o, c):
    return abs(c - o)


def _uptrend(df, i):
    ma = df["trend_ma"].iat[i]
    return not np.isnan(ma) and df["Close"].iat[i] > ma


def _package(name, trigger, stop, extra=None):
    """Turn a trigger and stop into a full setup, rejecting nonsensical risk."""
    if not np.isfinite(trigger) or not np.isfinite(stop) or stop >= trigger:
        return None
    risk = trigger - stop
    if risk / trigger < MIN_RISK_PCT or risk / trigger > MAX_RISK_PCT:
        return None
    out = {"signal": name, "trigger": float(trigger), "stop": float(stop),
           "target": float(trigger + REWARD_RISK * risk)}
    if extra:
        out.update(extra)
    return out


def pullback(df, i):
    """Uptrend, price dips back to the 20-day average, then turns up."""
    if i < TREND_MA or not _uptrend(df, i):
        return None
    lo = df["Low"].iloc[max(0, i - PULLBACK_LOOKBACK + 1):i + 1]
    ma = df["pull_ma"].iat[i]
    if np.isnan(ma) or lo.min() > ma * PULLBACK_TOUCH:
        return None
    if not (df["Close"].iat[i] > df["Open"].iat[i] and df["Close"].iat[i] > df["Close"].iat[i - 1]):
        return None
    return _package("pullback", df["High"].iat[i], df["Low"].iat[i])


def engulfing(df, i):
    """Yesterday red, today green and completely swallowing it."""
    if i < TREND_MA or not _uptrend(df, i):
        return None
    po, pc, o, c = (df["Open"].iat[i - 1], df["Close"].iat[i - 1],
                    df["Open"].iat[i], df["Close"].iat[i])
    prev_body, body = _body(po, pc), _body(o, c)
    if not (pc < po and c > o and prev_body > 0):
        return None
    if not (o <= pc and c >= po and body >= ENGULF_BODY_RATIO * prev_body):
        return None
    return _package("engulfing", df["High"].iat[i], min(df["Low"].iat[i], df["Low"].iat[i - 1]))


def hammer(df, i):
    """A long lower wick: sellers pushed it down, buyers took it back."""
    if i < TREND_MA or not _uptrend(df, i):
        return None
    o, h, l, c = (df["Open"].iat[i], df["High"].iat[i], df["Low"].iat[i], df["Close"].iat[i])
    body = _body(o, c)
    if body <= 0:
        return None
    lower_wick, upper_wick = min(o, c) - l, h - max(o, c)
    if lower_wick < HAMMER_WICK_RATIO * body or upper_wick > HAMMER_UPPER_WICK * body:
        return None
    ma = df["pull_ma"].iat[i]                      # only near support, not mid-rally
    if np.isnan(ma) or l > ma * HAMMER_NEAR_MA:
        return None
    return _package("hammer", h, l)


def piercing(df, i):
    """Opens below yesterday's close, then closes back above the middle of it."""
    if i < TREND_MA or not _uptrend(df, i):
        return None
    po, pc = df["Open"].iat[i - 1], df["Close"].iat[i - 1]
    o, c = df["Open"].iat[i], df["Close"].iat[i]
    prev_body = _body(po, pc)
    if not (pc < po and prev_body > 0 and c > o and o < pc):
        return None
    midpoint = pc + prev_body / 2
    if not (c > midpoint and c < po):
        return None
    return _package("piercing", df["High"].iat[i], min(df["Low"].iat[i], df["Low"].iat[i - 1]))


def morning_star(df, i):
    """Big red candle, a small hesitant candle, then a big green one."""
    if i < TREND_MA + 2 or not _uptrend(df, i):
        return None
    o1, c1 = df["Open"].iat[i - 2], df["Close"].iat[i - 2]
    o2, c2 = df["Open"].iat[i - 1], df["Close"].iat[i - 1]
    o3, c3 = df["Open"].iat[i], df["Close"].iat[i]
    b1, b2 = _body(o1, c1), _body(o2, c2)
    if not (c1 < o1 and b1 > 0 and b2 <= STAR_SMALL_BODY * b1 and c3 > o3):
        return None
    if c3 <= c1 + b1 / 2:
        return None
    stop = min(df["Low"].iat[i - 2], df["Low"].iat[i - 1], df["Low"].iat[i])
    return _package("morning_star", df["High"].iat[i], stop)


def breakout(df, i):
    """New 20-day closing high on at least double the usual volume."""
    if i < TREND_MA or not _uptrend(df, i):
        return None
    c, vol = df["Close"].iat[i], df["Volume"].iat[i]
    if np.isnan(df["close_high"].iat[i]) or c < df["close_high"].iat[i]:
        return None
    avg_vol = df["vol_avg"].iat[i]
    if np.isnan(avg_vol) or avg_vol <= 0 or vol < BREAKOUT_VOL_MULT * avg_vol:
        return None
    trigger = df["High"].iat[i]
    swing_low = df["Low"].iloc[max(0, i - 4):i + 1].min()
    atr_stop = trigger - BREAKOUT_ATR_MULT * df["atr"].iat[i]
    stop = max(swing_low, atr_stop) if np.isfinite(atr_stop) else swing_low
    return _package("breakout", trigger, stop)


def rsi_bounce(df, i):
    """Was heavily sold (RSI under 35) but is still in an uptrend, and is turning up."""
    if i < TREND_MA or not _uptrend(df, i):
        return None
    window = df["rsi"].iloc[max(0, i - RSI_LOOKBACK + 1):i + 1]
    if window.isna().any() or window.min() >= RSI_OVERSOLD:
        return None
    if not (df["rsi"].iat[i] > df["rsi"].iat[i - 1] and df["Close"].iat[i] > df["Open"].iat[i]):
        return None
    stop = df["Low"].iloc[max(0, i - RSI_LOOKBACK + 1):i + 1].min()
    return _package("rsi_bounce", df["High"].iat[i], stop)


DETECTORS = {f.__name__: f for f in
             (pullback, engulfing, hammer, piercing, morning_star, breakout, rsi_bounce)}


def detect_all(df, i):
    """Every setup firing on bar i. df must already have add_indicators() applied."""
    found = []
    for name, fn in DETECTORS.items():
        try:
            sig = fn(df, i)
        except (IndexError, KeyError):
            sig = None
        if sig:
            found.append(sig)
    return found

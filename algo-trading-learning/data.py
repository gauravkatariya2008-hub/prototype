"""
Getting daily prices and keeping a local copy.

--update re-downloads the full history every time, on purpose. Prices are
split- and dividend-adjusted, and a new split rewrites every older price. Stitching
fresh data onto an old cached copy would create fake crashes on split days, and the
backtest would happily trade them. So an update over the whole NSE list takes a
while (roughly 10-20 minutes); run it once a day after the market closes.

scan.py and evaluate.py only read the cache, so they are fast. Cached files live
in data_cache/ and are gitignored.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import signals
import universe

CACHE = Path(__file__).with_name("data_cache")
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
BATCH = 50


def _cache_path(symbol):
    return CACHE / f"{symbol.replace('/', '_')}.csv"


def read_cached(symbol):
    path = _cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df[COLUMNS].dropna() if set(COLUMNS).issubset(df.columns) else None


def update_cache(symbols, years=8, pause=0.4):
    """Download each symbol's full history and replace its cached copy."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance is not installed. Run:  pip install yfinance pandas")

    CACHE.mkdir(exist_ok=True)
    ok = failed = 0
    for start in range(0, len(symbols), BATCH):
        chunk = symbols[start:start + BATCH]
        tickers = [f"{s}.NS" for s in chunk]
        try:
            raw = yf.download(tickers, period=f"{years}y", interval="1d",
                              auto_adjust=True, progress=False, group_by="ticker",
                              threads=True)
        except Exception as exc:                        # noqa: BLE001
            print(f"  batch starting {chunk[0]} failed: {exc}")
            failed += len(chunk)
            continue

        for sym, ticker in zip(chunk, tickers):
            try:
                df = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[COLUMNS].dropna()
                if df.empty:
                    failed += 1
                    continue
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.index.name = "Date"
                df.to_csv(_cache_path(sym))
                ok += 1
            except (KeyError, ValueError):
                failed += 1
        print(f"  cached {ok} symbols, {failed} unavailable "
              f"({min(start + BATCH, len(symbols))}/{len(symbols)})")
        time.sleep(pause)
    return ok, failed


def demo_frames(n_symbols=12, years=8, seed=7):
    """Synthetic random-walk stocks, so everything is testable without internet."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=years * 250)
    frames = {}
    for k in range(n_symbols):
        drift = rng.normal(0.0003, 0.0004)
        close = 500 * np.exp(np.cumsum(rng.normal(drift, 0.016, len(days))))
        spread = close * rng.uniform(0.004, 0.012, len(days))
        open_ = close * (1 + rng.normal(0, 0.005, len(days)))
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        vol = rng.lognormal(13, 0.6, len(days))
        frames[f"DEMO{k + 1}"] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=days)
    return frames


def load_frames(symbols=None, years=8, demo=False, min_turnover_cr=None, quiet=False):
    """
    Price frames with indicators attached, ready for signals.detect_all().
    Reads only the cache — call update_cache() first to refresh it.
    """
    raw = demo_frames(years=years) if demo else {}
    if not demo:
        wanted = symbols or universe.load_symbols()
        for sym in wanted:
            df = read_cached(sym)
            if df is not None and not df.empty:
                raw[sym] = df
        if not raw and not quiet:
            print("Nothing in the cache yet. Run:  python scan.py --update")

    out = {}
    for sym, df in raw.items():
        with_ind = signals.add_indicators(df)
        if min_turnover_cr is not None and not universe.passes_filters(with_ind, min_turnover_cr):
            continue
        out[sym] = with_ind
    return out

"""
Which stocks to scan, and which to throw away.

The full NSE list (~2000 symbols) is fetched with --refresh-universe. NSE blocks
plain requests fairly often, so a built-in list of large, liquid names ships as a
fallback and the tool keeps working either way.

The liquidity filter matters more than it looks. A stock trading a few lakh
rupees a day will produce a beautiful backtest you can never achieve, because
when you try to sell there is nobody on the other side. Turnover is shown for
every pick so you can judge for yourself.
"""

import io
from pathlib import Path

import pandas as pd

SYMBOLS_FILE = Path(__file__).with_name("symbols.csv")
NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
MIN_TURNOVER_CR = 5.0        # rupees crore of median daily traded value
MIN_HISTORY_DAYS = 250

# Large, liquid NSE names — the fallback when the NSE download is unavailable.
FALLBACK = """
ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJAJFINSV BAJFINANCE
BEL BHARTIARTL BPCL BRITANNIA CIPLA COALINDIA DIVISLAB DRREDDY EICHERMOT ETERNAL
GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR ICICIBANK INDUSINDBK
INFY ITC JIOFIN JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID
RELIANCE SBILIFE SBIN SHRIRAMFIN SUNPHARMA TATACONSUM TATAMOTORS TATASTEEL TCS
TECHM TITAN TRENT ULTRACEMCO UPL WIPRO
ABB ADANIGREEN AMBUJACEM AUBANK BANKBARODA BERGEPAINT BIOCON BOSCHLTD CANBK
CHOLAFIN COLPAL DABUR DLF GAIL GODREJCP HAVELLS ICICIGI IDFCFIRSTB INDIGO
INDUSTOWER IOC IRCTC JINDALSTEL JUBLFOOD LICI LUPIN MARICO MOTHERSON MPHASIS
MUTHOOTFIN NAUKRI NMDC PAGEIND PEL PERSISTENT PFC PIDILITIND PIIND PNB POLYCAB
RECLTD SAIL SIEMENS SRF TATACHEM TATAELXSI TATAPOWER TORNTPHARM TVSMOTOR VEDL
VOLTAS ZYDUSLIFE
""".split()


def load_symbols(refresh=False):
    """The symbol list: NSE if asked and reachable, else the cached file, else FALLBACK."""
    if refresh:
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
            resp = requests.get(NSE_LIST_URL, headers=headers, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            # keep only the normal equity series; BE/BZ are restricted, illiquid segments
            if "SERIES" in df.columns:
                eq = df[df["SERIES"].astype(str).str.strip() == "EQ"]
            else:
                eq = df
            col = "SYMBOL" if "SYMBOL" in eq.columns else eq.columns[0]
            syms = sorted(eq[col].astype(str).str.strip().unique())
            SYMBOLS_FILE.write_text("\n".join(syms))
            print(f"Refreshed universe from NSE: {len(syms)} symbols.")
            refresh_industries()
            return syms
        except Exception as exc:                        # noqa: BLE001 - any failure falls back
            print(f"Could not refresh from NSE ({type(exc).__name__}: {exc}).")
            print("Using the cached list instead — the scan still works.")

    if SYMBOLS_FILE.exists():
        syms = [s.strip() for s in SYMBOLS_FILE.read_text().split() if s.strip()]
        if syms:
            return syms
    return list(FALLBACK)


def turnover_crores(df):
    """Median daily traded value over the last 60 days, in rupees crore."""
    if "turnover" not in df or df["turnover"].dropna().empty:
        return 0.0
    return float(df["turnover"].dropna().iloc[-1]) / 1e7


def passes_filters(df, min_turnover_cr=MIN_TURNOVER_CR):
    """True if this stock is liquid enough and has enough history to judge."""
    if len(df) < MIN_HISTORY_DAYS:
        return False
    return turnover_crores(df) >= min_turnover_cr


# --- industries (used to link sector news to your picks) ----------------------

INDUSTRIES_FILE = Path(__file__).with_name("industries.csv")
NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"


def refresh_industries():
    """Company name and NSE industry for the Nifty 500, saved to industries.csv."""
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
        resp = requests.get(NIFTY500_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        out = df.rename(columns={"Symbol": "symbol", "Company Name": "company",
                                 "Industry": "industry"})[["symbol", "company", "industry"]]
        out.to_csv(INDUSTRIES_FILE, index=False)
        print(f"Saved industries for {len(out)} companies.")
        return True
    except Exception as exc:                            # noqa: BLE001 - optional extra
        print(f"Could not fetch industries from NSE ({type(exc).__name__}). Sector news "
              f"links will be missing until it works.")
        return False


def load_industries():
    """{symbol: {"company": ..., "industry": ...}} or {} if never fetched."""
    if not INDUSTRIES_FILE.exists():
        return {}
    df = pd.read_csv(INDUSTRIES_FILE)
    return {r.symbol: {"company": r.company, "industry": r.industry}
            for r in df.itertuples(index=False)}

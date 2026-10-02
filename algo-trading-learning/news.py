"""
Market news as a WARNING SYSTEM, not a stock picker.

Collects headlines from Indian business newspapers, business TV channels'
websites and Google News searches on market-moving topics (RBI, oil, the
rupee, the US Fed, tariffs, wars). Each headline is tagged, by plain keyword
rules you can read below, with the NSE industries it can move. Headlines about
a specific company are checked for red flags (SEBI action, fraud, defaults,
results day).

Why it never picks stocks: by the time a headline reaches you, large funds have
already traded on it, and news-driven picks cannot be tested on past prices the
way every strategy in this lab is. Use it to avoid walking into trouble.

Not covered: live TV broadcasts (only the channels' web stories) and X/Twitter
(its data is paid and scraping it breaks its rules).
"""

import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd

GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

# Business newspapers and TV channels' websites. Any feed that fails is skipped.
PUBLISHER_FEEDS = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Mint Markets": "https://www.livemint.com/rss/markets",
    "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
    "Business Standard Markets": "https://www.business-standard.com/rss/markets-106.rss",
}

# Google News searches for the big drivers of Indian stock prices. These pull in
# newspapers, TV channels' websites (CNBC-TV18, ET Now, NDTV Profit) and wires.
TOPIC_SEARCHES = {
    "Indian market today": "Sensex Nifty today",
    "RBI and interest rates": "RBI repo rate policy",
    "Crude oil": "crude oil price India",
    "Rupee": "rupee dollar exchange rate",
    "US Federal Reserve": "US Federal Reserve rates markets",
    "Trade and tariffs": "India tariffs trade deal",
    "Geopolitics": "geopolitical tensions India markets",
    "Foreign investors": "FII FPI flows India stocks",
    "Government and budget": "India government policy stocks GST budget",
}

ALL = "All stocks"

# (keywords, NSE industries the news can move, why). Industries use NSE's names.
SECTOR_RULES = [
    (["crude", "oil price", "brent", "opec", "petrol", "diesel"],
     ["Oil Gas & Consumable Fuels", "Chemicals", "Automobile and Auto Components", "Services"],
     "oil is a major cost or revenue for these (fuel, paints, tyres, airlines)"),
    (["rupee", "forex", "dollar index"],
     ["Information Technology", "Healthcare", "Oil Gas & Consumable Fuels"],
     "exporters earn in dollars; importers pay in dollars"),
    (["repo rate", "rate cut", "rate hike", "interest rate", "rbi", "inflation", "liquidity"],
     ["Financial Services", "Realty", "Automobile and Auto Components"],
     "borrowing costs drive banks, lenders, property and vehicle loans"),
    (["federal reserve", "fed ", "us yields", "treasury yield", "powell"],
     [ALL, "Information Technology"],
     "US rates move foreign money in and out of India and US tech spending"),
    (["tariff", "trade war", "trade deal", "sanction", "export ban", "import duty", "anti-dumping"],
     ["Metals & Mining", "Textiles", "Chemicals", "Automobile and Auto Components",
      "Information Technology"],
     "trade rules change what exporters can sell and at what price"),
    (["war", "conflict", "missile", "attack", "border", "tension", "ceasefire", "strike on"],
     [ALL, "Capital Goods", "Oil Gas & Consumable Fuels"],
     "conflict hits the whole market, oil prices and defence companies"),
    (["monsoon", "rainfall", "kharif", "rabi", "drought", "heatwave"],
     ["Fast Moving Consumer Goods", "Chemicals", "Automobile and Auto Components"],
     "rural demand, fertilisers and tractors depend on the rains"),
    (["steel", "iron ore", "aluminium", "copper", "zinc", "metal prices"],
     ["Metals & Mining", "Capital Goods", "Construction"],
     "metal prices are revenue for miners and cost for builders"),
    (["gold price", "gold rate", "silver price"],
     ["Consumer Durables", "Financial Services"],
     "jewellers and gold-loan lenders"),
    (["usfda", "drug price", "pharma"],
     ["Healthcare"], "US drug approvals and pricing drive pharma exporters"),
    (["telecom", "tariff hike", "spectrum", "5g"],
     ["Telecommunication"], "telecom pricing and spectrum costs"),
    (["power demand", "electricity", "coal supply", "renewable", "solar"],
     ["Power"], "power demand and fuel supply"),
    (["cement", "infrastructure", "capex", "road project", "railway"],
     ["Construction Materials", "Construction", "Capital Goods"],
     "government and private building spend"),
    (["fii", "fpi", "foreign investors", "foreign outflow", "foreign inflow"],
     [ALL], "foreign money flows move the whole market"),
    (["budget", "gst", "income tax", "capital gains tax", "stt"],
     [ALL], "tax changes affect every investor"),
]

# Headlines about one company that mean "look before you buy".
RED_FLAGS = {
    "regulator": ["sebi", "ed raid", "income tax raid", "probe", "investigation", "show cause",
                  "penalty", "ban"],
    "fraud or governance": ["fraud", "auditor resign", "resigns", "whistleblower", "forensic",
                            "pledge"],
    "money trouble": ["default", "insolvency", "nclt", "downgrade", "debt restructuring",
                      "missed payment"],
    "results": ["q1 results", "q2 results", "q3 results", "q4 results", "quarterly results",
                "earnings", "net profit", "board meeting"],
}
RED_FLAG_NOTE = {
    "regulator": "Regulator or investigator involved",
    "fraud or governance": "Possible governance problem",
    "money trouble": "Possible financial stress",
    "results": "Results news: the price can swing sharply around results",
}


def _has(text, word):
    return re.search(r"(?<![a-z])" + re.escape(word.strip()) + r"(?![a-z])", text) is not None


def tag_sectors(title):
    """[(industry, why)] the headline can move, from SECTOR_RULES."""
    text = title.lower()
    out, seen = [], set()
    for words, industries, why in SECTOR_RULES:
        if any(_has(text, w) for w in words):
            for ind in industries:
                if ind not in seen:
                    seen.add(ind)
                    out.append((ind, why))
    return out


def red_flags(title):
    text = title.lower()
    return [kind for kind, words in RED_FLAGS.items() if any(_has(text, w) for w in words)]


def parse_feed(xml_text, source=None):
    """Headlines from RSS XML as dicts: title, link, published, source."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        publisher = item.findtext("source") or source or ""
        if not item.findtext("source") and " - " in title and source is None:
            title, publisher = title.rsplit(" - ", 1)
        elif item.findtext("source") and title.endswith(" - " + publisher):
            title = title[: -len(" - " + publisher)]
        published = None
        raw_date = item.findtext("pubDate")
        if raw_date:
            try:
                published = pd.Timestamp(parsedate_to_datetime(raw_date)).tz_convert(
                    "Asia/Kolkata")
            except (TypeError, ValueError):
                published = None
        items.append({"title": title, "link": (item.findtext("link") or "").strip(),
                      "published": published, "source": publisher.strip()})
    return items


def fetch(url, timeout=10):
    import requests
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text


def _gather(jobs, fetcher):
    """jobs: [(label, url, source or None)]. Returns (headlines, failed labels)."""
    def one(job):
        label, url, source = job
        try:
            return label, parse_feed(fetcher(url), source), None
        except Exception as exc:                        # noqa: BLE001 - one bad feed is fine
            return label, [], f"{label} ({type(exc).__name__})"
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, jobs))
    rows, failed = [], []
    for label, items, err in results:
        if err:
            failed.append(err)
        for it in items:
            rows.append({**it, "topic": label})
    return rows, failed


def _frame(rows, max_age_days):
    df = pd.DataFrame(rows, columns=["title", "link", "published", "source", "topic"])
    if df.empty:
        return df
    df = df.drop_duplicates("title")
    if max_age_days is not None and df["published"].notna().any():
        cutoff = pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=max_age_days)
        df = df[df["published"].isna() | (df["published"] >= cutoff)]
    return df.sort_values("published", ascending=False, na_position="last").reset_index(drop=True)


def market_news(fetcher=fetch, max_age_days=3):
    """Market-wide and geopolitical headlines, each tagged with the industries it can move."""
    jobs = [(label, GOOGLE_NEWS.format(q=quote_plus(q)), None) for label, q in TOPIC_SEARCHES.items()]
    jobs += [(name, url, name) for name, url in PUBLISHER_FEEDS.items()]
    rows, failed = _gather(jobs, fetcher)
    df = _frame(rows, max_age_days)
    if not df.empty:
        df["sectors"] = df["title"].map(lambda t: [s for s, _ in tag_sectors(t)])
    return df, failed


def company_news(symbol, company=None, fetcher=fetch, max_age_days=14):
    """Recent headlines about one company, with red flags marked."""
    name = company or symbol
    query = f'"{name}" share OR stock' if company else f"{symbol} NSE share"
    rows, failed = _gather([(symbol, GOOGLE_NEWS.format(q=quote_plus(query)), None)], fetcher)
    df = _frame(rows, max_age_days)
    if not df.empty:
        df["flags"] = df["title"].map(red_flags)
    return df, failed


def news_for_industry(market, industry):
    """Market headlines tagged with this industry or with the whole market."""
    if market is None or market.empty or "sectors" not in market:
        return market.iloc[0:0] if market is not None else pd.DataFrame()
    mask = market["sectors"].map(lambda s: industry in s or ALL in s)
    return market[mask]

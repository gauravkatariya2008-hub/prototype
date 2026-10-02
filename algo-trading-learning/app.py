"""
NSE Strategy Research Lab — a dashboard for testing trading ideas honestly.

Run it with:
    pip install streamlit
    streamlit run app.py

It is RESEARCH ONLY. It never places orders, never asks for a Zerodha login,
and has no code that could. Every number is arithmetic on past prices.
"""

import json
import os

import altair as alt
import pandas as pd
import streamlit as st

import data
import evaluate
import news
import picks
import portfolio
import scan
import strategies
import universe

st.set_page_config(page_title="NSE Strategy Research Lab", page_icon="📊", layout="wide")
alt.data_transformers.disable_max_rows()

# Two-series palette (validated for colour-blind separation and contrast, light and dark).
PALETTE = {"light": ("#2a78d6", "#eb6834"), "dark": ("#3987e5", "#d95926")}
LABEL_INK = {"light": "#52514e", "dark": "#c3c2b7"}       # text colour, never the series colour
STRAT, BENCH = "Strategy", "NIFTYBEES"


def theme_mode():
    try:
        return st.context.theme.type or "light"
    except Exception:  # noqa: BLE001 - older Streamlit or no browser context
        return "light"


def colors():
    return PALETTE.get(theme_mode(), PALETTE["light"])


def points(v):
    """Percentage-point difference, without a misleading '-0.0'."""
    v = round(v * 100, 1)
    return f"{(0.0 if v == 0 else v):+.1f} pts"


def rupees(v):
    return f"₹{v:,.0f}"


def pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:+.1%}"


# ----------------------------------------------------------------- data, cached

@st.cache_data(show_spinner="Loading prices… (first load of the full NSE list can take a minute)")
def get_panel(demo):
    return data.load_panel(demo=demo)


@st.cache_data(show_spinner="Running the backtest…")
def get_backtest(demo, key, capital, top_n, min_turnover, start, end, split):
    panel = get_panel(demo)
    return portfolio.run(panel, strategies.STRATEGIES[key], capital=capital, top_n=top_n,
                         start=start, end=end, min_turnover_cr=min_turnover, split=split)


@st.cache_data(show_spinner="Checking today's setups…")
def get_frames(demo, min_turnover):
    return data.load_frames(demo=demo, min_turnover_cr=min_turnover, quiet=True)


def load_stats():
    if not evaluate.STATS_FILE.exists():
        return None
    try:
        return json.loads(evaluate.STATS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ----------------------------------------------------------------------- charts

def two_line_chart(df, y_title, fmt, height=320):
    """Strategy vs NIFTYBEES over time, one axis, crosshair tooltip, labelled lines."""
    c_strat, c_bench = colors()
    long = df.reset_index(names="date").melt("date", var_name="series", value_name="value")
    color = alt.Color("series:N", scale=alt.Scale(domain=[STRAT, BENCH], range=[c_strat, c_bench]),
                      legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(long).encode(x=alt.X("date:T", title=None,
                                           axis=alt.Axis(format="%Y", tickCount="year")))
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False,
                                clear="pointerout")
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=fmt)), color=color)
    dots = base.mark_point(size=64, filled=True).encode(
        y="value:Q", color=color, opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    rule = base.transform_pivot("series", value="value", groupby=["date"]).mark_rule(
        color="#8a8984").encode(
        opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip(f"{STRAT}:Q", title=STRAT, format=fmt),
                 alt.Tooltip(f"{BENCH}:Q", title=BENCH, format=fmt)]).add_params(hover)
    last = long[long["date"] == long["date"].max()]
    labels = alt.Chart(last).mark_text(align="left", dx=6, fontSize=12,
                                       color=LABEL_INK.get(theme_mode(), "#52514e")).encode(
        x="date:T", y="value:Q", text="series:N")
    return (lines + dots + rule + labels).properties(
        height=height, padding={"left": 5, "top": 5, "bottom": 5, "right": 80})


def yearly_chart(yearly):
    c_strat, c_bench = colors()
    long = yearly.melt(id_vars=["year"], value_vars=["strategy", "benchmark"],
                       var_name="series", value_name="return")
    long["series"] = long["series"].map({"strategy": STRAT, "benchmark": BENCH})
    return alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("year:O", title=None, scale=alt.Scale(paddingInner=0.25),
                axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("series:N", scale=alt.Scale(domain=[STRAT, BENCH], paddingInner=0.08)),
        y=alt.Y("return:Q", title="Return in the year", axis=alt.Axis(format="%")),
        color=alt.Color("series:N", scale=alt.Scale(domain=[STRAT, BENCH],
                                                    range=[c_strat, c_bench]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("return:Q", title="Return", format="+.1%")],
    ).properties(height=280)


# ---------------------------------------------------------------------- sidebar

st.sidebar.title("Research Lab")
st.sidebar.caption("Research only. This app never places orders and never asks for "
                   "your Zerodha login.")

real_panel_available = any(data.CACHE.glob("*.csv")) if data.CACHE.exists() else False
source = st.sidebar.radio(
    "Price data", ["My downloaded prices", "Demo (made-up prices)"],
    index=0 if real_panel_available and not os.environ.get("LAB_FORCE_DEMO") else 1,
    help="Download real prices first with:  python scan.py --update --refresh-universe")
demo = source.startswith("Demo")
if not demo and not real_panel_available:
    st.sidebar.warning("No downloaded prices found, so the demo data is being used. Run "
                       "`python scan.py --update --refresh-universe` first.")
    demo = True

your_capital = st.sidebar.number_input("Your budget (₹)", min_value=500, value=2000, step=500,
                                       help="What you would actually invest. Today's picks are "
                                            "planned for this amount, and every page shows what "
                                            "happens at this size.")
min_turnover = st.sidebar.number_input("Minimum traded value (₹ crore/day)", min_value=0.0,
                                       value=universe.MIN_TURNOVER_CR, step=1.0,
                                       help="Skip stocks that trade less than this. Thin stocks "
                                            "look great in backtests and are hard to sell in "
                                            "real life.")
if demo:
    st.sidebar.info("Demo mode: random made-up prices. Use it to learn the app; ignore the "
                    "results.")

st.title("NSE Strategy Research Lab")
st.caption("Tests trading ideas on past NSE prices with Zerodha charges, and asks one "
           "question: does it beat simply buying NIFTYBEES and holding it?")

tab_picks, tab_lab, tab_rules, tab_scan, tab_limits = st.tabs(
    ["Today's picks", "Strategy Lab", "Candlestick rules", "Today's scan", "Limits"])

# ------------------------------------------------------------------ strategy lab

# --------------------------------------------------------------- today's picks

@st.cache_data(show_spinner="Testing the 5-stock version of each strategy on your data…")
def get_pick_tests(demo, min_turnover):
    panel = get_panel(demo)
    out = {}
    for key, strat in strategies.STRATEGIES.items():
        try:
            out[key] = portfolio.run(panel, strat, capital=500000.0,
                                     top_n=picks.tested_top_n(key), min_turnover_cr=min_turnover)
        except ValueError:
            continue
    return out


@st.cache_data(ttl=1800, show_spinner="Reading today's news…")
def get_market_news():
    return news.market_news()


@st.cache_data(ttl=1800, show_spinner=False)
def get_company_news(symbol, company):
    return news.company_news(symbol, company)


def md_escape(text):
    for ch in "\\[]*_`#<>":
        text = text.replace(ch, "\\" + ch)
    return text


def age(ts):
    if ts is None or pd.isna(ts):
        return ""
    mins = (pd.Timestamp.now(tz="Asia/Kolkata") - ts).total_seconds() / 60
    if mins < 60:
        return f"{max(1, int(mins))} min ago"
    if mins < 60 * 24:
        return f"{int(mins // 60)}h ago"
    return f"{int(mins // (60 * 24))}d ago"


def headline(row, tags=None):
    when = age(row["published"])
    meta = ", ".join(x for x in (row["source"], when) if x)
    tag = f" · _{', '.join(tags)}_" if tags else ""
    return f"- [{md_escape(row['title'])}]({row['link']}) — {meta}{tag}"


def next_review(key, last_day):
    strat = strategies.STRATEGIES[key]
    if strat.rebalance == "M":
        return f"the last trading day of {last_day.strftime('%B %Y')}"
    return "the last trading day of this week"


def render_picks():
    st.subheader("Today's picks")
    st.caption("Suggestions only — this app never places orders. Picks come only from a "
               "strategy that beat NIFTYBEES on your own price history.")
    panel = get_panel(demo)
    if panel is None or data.BENCHMARK not in panel["close"].columns:
        st.error(f"{data.BENCHMARK} prices are missing. Download them with:  "
                 f"`python scan.py --update --symbols {data.BENCHMARK}`")
        return
    last_day = panel["close"].index[-1]
    tests = get_pick_tests(demo, float(min_turnover))
    best, rows = picks.choose_strategy(tests)

    if best:
        b = next(r for r in rows if r["key"] == best)
        st.success(f"✅ Using **{b['label']}**: its 5-stock version beat NIFTYBEES on your data "
                   f"({pct(b['cagr'])} vs {pct(b['bench_cagr'])} a year, confidence "
                   f"{b['t']:.1f}). Past results only — not a promise.")
    else:
        st.warning("⚠️ **No strategy beat NIFTYBEES** on your data in its 5-stock version, so "
                   "there are no stock picks worth trusting today. The honest pick is "
                   "NIFTYBEES itself.")
    st.dataframe(pd.DataFrame([{
        "Strategy (5 stocks)": r["label"],
        "Verdict": "Beats NIFTYBEES" if r["beats"] else "Does not beat",
        "Yearly return": pct(r["cagr"]), "NIFTYBEES": pct(r["bench_cagr"]),
        "Confidence (need 2+)": "n/a" if pd.isna(r["t"]) else f"{r['t']:.1f}",
        "Main reason": r["reasons"][0] if r["reasons"] else "passed all three tests",
    } for r in rows]), hide_index=True, width="stretch")

    industries = universe.load_industries()
    top = []
    if best:
        n_tested = picks.tested_top_n(best)
        ranked = picks.current_picks(panel, best, float(min_turnover), n=40)
        keep, top = ranked[:n_tested], ranked[:picks.PICKS]
        prices = picks.latest_prices(panel, ranked + [data.BENCHMARK])
        st.markdown(f"### Top {len(top)} right now")
        turnover = (panel["close"] * panel["volume"]).rolling(60, min_periods=40).median().iloc[-1]
        st.dataframe(pd.DataFrame([{
            "Rank": i + 1, "Stock": s,
            "Company": industries.get(s, {}).get("company", ""),
            "Industry": industries.get(s, {}).get("industry", "unknown"),
            "Last close (₹)": round(prices.get(s, float("nan")), 2),
            "Traded value (₹ cr/day)": round(float(turnover.get(s, 0)) / 1e7, 1),
        } for i, s in enumerate(top)]), hide_index=True, width="stretch")
        st.caption(f"Based on prices up to {last_day.date()}. "
                   f"{strategies.STRATEGIES[best].label} reviews its list on "
                   f"{next_review(best, last_day)}: check back then for what to sell.")
    else:
        ranked, keep = [], None
        prices = picks.latest_prices(panel, [data.BENCHMARK])

    st.markdown(f"### Plan for your budget: {rupees(your_capital)}")
    plan_rows, left, note = picks.plan_budget(float(your_capital), ranked, prices)
    st.write(note)
    if plan_rows:
        st.dataframe(pd.DataFrame([{
            "Buy": r["symbol"], "Price (₹)": round(r["price"], 2), "Quantity": r["qty"],
            "Cost": rupees(r["cost"]), "Buy charges": f"₹{r['buy_charges']:,.2f}",
            "Share of budget": f"{r['share_of_budget']:.0%}",
        } for r in plan_rows]), hide_index=True, width="stretch")
        st.caption(f"Left over in cash: {rupees(left)}. Place these yourself in Kite as "
                   f"Delivery (Longterm) orders.")
        if best and plan_rows[0]["symbol"] != data.BENCHMARK and len(plan_rows) < len(top):
            st.warning(f"Your budget holds {len(plan_rows)} of the {len(top)} picks. The tested "
                       f"version held {len(top)}, so your results can differ a lot from the "
                       f"backtest.")

    st.markdown("### Check what you already hold")
    st.caption("Type in what you bought. It stays on this page; nothing is sent anywhere.")
    held = st.data_editor(
        pd.DataFrame({"symbol": pd.Series(dtype="str"), "qty": pd.Series(dtype="float"),
                      "buy_price": pd.Series(dtype="float")}),
        num_rows="dynamic", key="holdings", width="stretch",
        column_config={"symbol": st.column_config.TextColumn("Stock (e.g. TCS)"),
                       "qty": st.column_config.NumberColumn("Quantity", min_value=0, step=1),
                       "buy_price": st.column_config.NumberColumn("Buy price (₹)", min_value=0)})
    held = held.dropna(how="all")
    if len(held):
        syms = [str(x).strip().upper().removesuffix(".NS") for x in held["symbol"].dropna()]
        review = picks.review_holdings(held, picks.latest_prices(panel, syms + [data.BENCHMARK]),
                                       keep, best)
        if len(review):
            st.dataframe(review.assign(
                price=review["price"].map(lambda v: "—" if pd.isna(v) else f"₹{v:,.2f}"),
                value=review["value"].map(lambda v: "—" if pd.isna(v) else rupees(v)),
                profit_after_sell_charges=review["profit_after_sell_charges"].map(
                    lambda v: "—" if pd.isna(v) else f"₹{v:+,.0f}")).rename(columns={
                        "symbol": "Stock", "qty": "Quantity", "buy_price": "Bought at",
                        "price": "Now", "value": "Worth now",
                        "profit_after_sell_charges": "Profit if sold (after charges)",
                        "action": "What the strategy says"}),
                hide_index=True, width="stretch")

    st.markdown("### News that can move prices")
    st.caption("Newspapers, business TV channels' websites and searches on market-moving topics "
               "(RBI, oil, rupee, US Fed, tariffs, conflicts). A warning system, not a stock "
               "picker: by the time news reaches you, big funds have already traded on it. "
               "X/Twitter and live TV broadcasts are not included.")
    if not st.toggle("Load today's news (fetches from the internet)", key="news_on"):
        return
    market, failed = get_market_news()
    if market.empty:
        st.warning("Couldn't load any news. Check your internet connection and try again.")
        return
    if top and not demo:
        st.markdown("#### News about your picks")
        for sym in top:
            info = industries.get(sym, {})
            company, industry = info.get("company"), info.get("industry")
            cn, _ = get_company_news(sym, company)
            flagged = cn[cn["flags"].map(bool)] if len(cn) else cn
            sector = news.news_for_industry(market, industry) if industry else market.iloc[0:0]
            title = (f"{'⚠️ ' if len(flagged) else ''}{sym}"
                     f"{f' — {company}' if company else ''} · {len(cn)} company headlines, "
                     f"{len(sector)} market/sector headlines")
            with st.expander(title):
                for _, row in flagged.iterrows():
                    notes = "; ".join(news.RED_FLAG_NOTE[f] for f in row["flags"])
                    st.warning(f"**{notes}:** {row['title']}")
                st.markdown("**About the company**")
                st.markdown("\n".join(headline(r) for _, r in cn.head(6).iterrows())
                            or "_No recent headlines found._")
                if industry:
                    st.markdown(f"**Market and {industry} news**")
                    st.markdown("\n".join(headline(r, r["sectors"]) for _, r in
                                          sector.head(6).iterrows()) or "_Nothing recent._")
                else:
                    st.caption("Industry unknown — run `python scan.py --refresh-universe` to "
                               "fetch industries.")
    elif demo:
        st.info("Company news needs your downloaded prices (demo stocks are made up). The "
                "market news below is real.")
    st.markdown("#### Market and world news")
    st.markdown("\n".join(headline(r, r["sectors"]) for _, r in market.head(30).iterrows()))
    if failed:
        st.caption("Could not reach: " + ", ".join(failed))


with tab_picks:
    render_picks()


def render_lab():
    panel = get_panel(demo)
    if panel is None or data.BENCHMARK not in panel["close"].columns:
        st.error(f"{data.BENCHMARK} prices are missing, so there is nothing to compare "
                 f"against. Download them with:  `python scan.py --update --symbols "
                 f"{data.BENCHMARK}`")
        return

    first_day = panel["close"].index[0].date()
    last_day = panel["close"].index[-1].date()
    labels = {s.label: k for k, s in strategies.STRATEGIES.items()}

    with st.form("lab"):
        c1, c2 = st.columns([2, 3])
        with c1:
            label = st.selectbox("Strategy", list(labels))
            key = labels[label]
            strat = strategies.STRATEGIES[key]
            st.caption(strat.description)
        with c2:
            a, b, c = st.columns(3)
            capital = a.number_input("Research capital (₹)", min_value=10000, value=500000,
                                     step=50000, help="Judge the idea at a size where "
                                     "charges don't swamp it. Your own size is shown below.")
            top_n = b.number_input("Stocks held", min_value=1, max_value=50, value=strat.top_n,
                                   disabled=strat.benchmark_only)
            split = c.date_input("Split date", value=pd.Timestamp(portfolio.SPLIT_DATE).date(),
                                 min_value=first_day, max_value=last_day,
                                 help="The strategy must beat NIFTYBEES both before and after "
                                      "this date.")
            period = st.slider("Period", min_value=first_day, max_value=last_day,
                               value=(first_day, last_day), format="YYYY-MM-DD")
        st.form_submit_button("Run backtest", type="primary")

    args = dict(demo=demo, key=key, top_n=int(top_n), min_turnover=float(min_turnover),
                start=str(period[0]), end=str(period[1]), split=str(split))
    try:
        res = get_backtest(capital=float(capital), **args)
        small = get_backtest(capital=float(your_capital), **args)
    except ValueError as exc:
        st.warning(f"{exc} Pick a longer period.")
        return

    m, v = res.metrics, res.verdict
    if v["beats_benchmark"]:
        st.success(f"✅ **Beats NIFTYBEES** on all three tests ({label}, "
                   f"{m['start']} to {m['end']}). Past results only — paper trade it before "
                   f"believing it.")
    else:
        st.error(f"❌ **Does not beat NIFTYBEES** ({label}, {m['start']} to {m['end']})")
        for r in v["reasons"]:
            st.markdown(f"- {r}")

    def versus(diff):
        """(delta text, delta colour) for a percentage-point gap vs NIFTYBEES."""
        if abs(diff) < 0.0005:
            return "level with NIFTYBEES", "off"
        return f"{points(diff)} vs NIFTYBEES", "normal"

    k1, k2, k3 = st.columns(3)
    k4, k5, k6 = st.columns(3)
    txt, tone = versus(m["cagr"] - m["bench_cagr"])
    k1.metric("Yearly return", pct(m["cagr"]), txt, delta_color=tone,
              delta_arrow="off" if tone == "off" else "auto",
              help=f"NIFTYBEES: {pct(m['bench_cagr'])} a year")
    txt, tone = versus(m["max_drawdown"] - m["bench_max_drawdown"])
    k2.metric("Worst fall from peak", pct(m["max_drawdown"]), txt, delta_color=tone,
              delta_arrow="off" if tone == "off" else "auto",
              help=f"NIFTYBEES: {pct(m['bench_max_drawdown'])}. A smaller fall is better.")
    k3.metric("Final value", rupees(m["final_value"]),
              f"{rupees(m['final_value'] - m['bench_final'])} vs NIFTYBEES")
    k4.metric(f"Charges paid ({m['charges_paid'] / capital:.1%} of capital)",
              rupees(m["charges_paid"]))
    k5.metric("Trades", f"{m['trades']:,}")
    k6.metric("Confidence (need 2+)", "n/a" if pd.isna(v["t"]) else f"{v['t']:.1f}",
              help="How many standard errors the monthly lead over NIFTYBEES is from zero. "
                   "Below 2, the difference could easily be luck.")

    st.subheader("Value over time")
    st.altair_chart(two_line_chart(
        pd.DataFrame({STRAT: res.equity, BENCH: res.benchmark}), "Value (₹)", ",.0f"),
        width="stretch")

    st.subheader("Falls from the previous peak")
    dd = pd.DataFrame({STRAT: res.equity / res.equity.cummax() - 1,
                       BENCH: res.benchmark / res.benchmark.cummax() - 1})
    st.altair_chart(two_line_chart(dd, "Below previous peak", ".0%", height=240),
                    width="stretch")

    st.subheader("Year by year")
    full = res.yearly[~res.yearly["partial"]]
    st.caption(f"Beat NIFTYBEES in {v['years_beaten']} of {v['full_years']} full years. "
               f"Partial first and last years are shown but not counted.")
    st.altair_chart(yearly_chart(res.yearly), width="stretch")
    table = res.yearly.assign(
        Year=res.yearly["year"].astype(str) + res.yearly["partial"].map({True: " (partial)",
                                                                           False: ""}),
        Strategy=res.yearly["strategy"].map(pct), NIFTYBEES=res.yearly["benchmark"].map(pct),
        Beat=res.yearly["beat"].map({True: "yes", False: "no"}))[["Year", "Strategy",
                                                                   "NIFTYBEES", "Beat"]]
    st.dataframe(table, hide_index=True, width="stretch")
    del full

    st.subheader(f"At your capital: {rupees(your_capital)}")
    sm = small.metrics
    lead = sm["final_value"] - sm["bench_final"]
    st.markdown(
        f"- **Final value:** {rupees(sm['final_value'])} vs {rupees(sm['bench_final'])} for "
        f"NIFTYBEES ({'ahead' if lead > 0 else 'behind'} by {rupees(abs(lead))})\n"
        f"- **Charges paid:** {rupees(sm['charges_paid'])} on {sm['trades']:,} trades\n"
        f"- **Stocks it wanted but you couldn't afford:** {sm['unaffordable']:,} of "
        f"{sm['buy_slots']:,} buys")
    if sm["buy_slots"] and sm["unaffordable"] / sm["buy_slots"] > 0.3:
        st.warning("At this size you can't hold what the strategy picks, so you are not really "
                   "running this strategy. Results here say little about the idea itself.")

    with st.expander(f"All {len(res.trades):,} trades at research capital"):
        st.dataframe(res.trades, hide_index=True, width="stretch")
        st.download_button("Download trades (CSV)", res.trades.to_csv(index=False),
                           file_name=f"{key}_trades.csv", mime="text/csv")


with tab_lab:
    render_lab()

# -------------------------------------------------------------- candlestick rules

with tab_rules:
    st.subheader("Candlestick rules, measured")
    raw = load_stats()
    if not raw:
        st.info("No measurements yet. Run `python evaluate.py` in Command Prompt, then reload "
                "this page.")
    else:
        notional = raw.get("notional", 10000)
        rows = []
        for name, s in sorted(raw["signals"].items(), key=lambda kv: -kv[1]["expectancy_rupees"]):
            if not s["reliable"]:
                verdict = "Too few trades — ignore"
            elif s["expectancy_rupees"] <= 0:
                verdict = "Lost money"
            elif s.get("robust"):
                verdict = "Made money and passed the stress test"
            elif "robust" in s:
                verdict = "Made money but failed the stress test"
            else:
                verdict = "Made money (stress test not run yet)"
            rows.append({"Rule": name, "Trades": s["samples"], "Won": f"{s['win_rate']:.0%}",
                         "Avg win": rupees(s["avg_win_rupees"]),
                         "Avg loss": rupees(abs(s["avg_loss_rupees"])),
                         "Per trade after charges": f"₹{s['expectancy_rupees']:+,.0f}",
                         "Verdict": verdict})
        st.caption(f"Each rule replayed on past prices at about {rupees(notional)} per trade, "
                   f"after Zerodha charges.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        stressed = {n: s["stress"] for n, s in raw["signals"].items() if "stress" in s}
        for name, sres in stressed.items():
            ok = sres["robust"]
            with st.expander(f"{'✅' if ok else '❌'} {name}: "
                             f"{'holds up' if ok else 'fragile'}"):
                st.markdown(
                    f"- Before {evaluate.SPLIT_DATE}: ₹{sres['early']['per_trade'] or 0:+,.0f} "
                    f"per trade ({sres['early']['samples']:,} trades)\n"
                    f"- From {evaluate.SPLIT_DATE}: ₹{sres['late']['per_trade'] or 0:+,.0f} "
                    f"per trade ({sres['late']['samples']:,} trades)\n"
                    f"- Profitable in {sres['positive_years']} of {sres['years_counted']} years\n"
                    f"- Confidence {sres['t_clustered']:.1f} (need 2+)")
                for reason in sres["reasons"]:
                    st.markdown(f"  - ❌ {reason}")

# ------------------------------------------------------------------ today's scan

with tab_scan:
    st.subheader("Today's candlestick setups")
    raw = load_stats()
    stats = raw.get("signals", {}) if raw else {}
    show_all = st.checkbox("Also show rules that failed testing (for learning only)")
    frames = get_frames(demo, float(min_turnover))
    frames, stale, latest = scan.fresh_frames(frames)
    if not frames:
        st.info("No price data. Run `python scan.py --update` first.")
    else:
        found, hidden = scan.find_setups(frames, stats, include_unproven=show_all)
        st.caption(f"{len(frames):,} stocks checked for {latest.date()}."
                   + (f" {stale} skipped with stale data." if stale else "")
                   + (f" {hidden} setups hidden because their rule failed testing."
                      if hidden else ""))
        if not stats:
            st.warning("These rules have not been measured yet (run `python evaluate.py`). "
                       "Nothing here is a recommendation.")
        if not found:
            st.info("Nothing qualifies today. That's the correct answer on most days, and "
                    "right now no candlestick rule has passed testing at all.")
        else:
            rows = []
            for _, sym, df, sigs in found[:25]:
                primary = max(sigs, key=lambda x: stats.get(x["signal"], {}).get(
                    "expectancy_r", -99))
                qty = scan.position_size(primary["trigger"], primary["stop"], your_capital, 0.01)
                rows.append({"Stock": sym, "Rules": " + ".join(x["signal"] for x in sigs),
                             "Buy above": round(primary["trigger"], 2),
                             "Stop": round(primary["stop"], 2),
                             "Target": round(primary["target"], 2),
                             f"Qty at {rupees(your_capital)}": qty or "paper trade only",
                             "Proven rule": "yes" if any(scan.proven(x["signal"], stats)
                                                         for x in sigs) else "no",
                             "Traded value (₹ cr/day)": round(universe.turnover_crores(df), 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("This page suggests only. It places no orders.")

# ------------------------------------------------------------------------ limits

with tab_limits:
    st.subheader("What this lab can and cannot tell you")
    st.markdown("""
**What it does honestly**
- Decisions use prices up to a day's close; trades happen at the **next** day's open.
- Every buy and sell pays Zerodha delivery charges, including the flat ₹15 DP charge per stock sold. ETFs are charged at their lower tax rate.
- Liquidity is judged on the 60 days *before* each decision, not on today's volume.
- A strategy's race against NIFTYBEES starts only once it has enough history to decide, so it never sits in cash while the benchmark is invested.
- To "beat NIFTYBEES" a strategy must do better **before and after** the split date, be **confident** (monthly results at least 2 standard errors ahead) and beat it in **60%+ of full years**.

**What it cannot do**
- **Predict the future.** Every number is historical.
- **Include delisted companies.** Price history only exists for stocks still listed, so failures are missing and results look better than reality. Momentum is especially flattered by this.
- **Count income tax.** Frequent trading pays 20% short-term capital gains tax; holding NIFTYBEES over a year pays 12.5% above ₹1.25 lakh. That gap favours buy-and-hold even more than these numbers show.
- **Place orders.** There is no trading code in this app at all.
""")

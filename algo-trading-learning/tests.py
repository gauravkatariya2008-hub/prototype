"""
Tests for the scanner. Run:  python tests.py

These matter more than usual. A bug here does not crash anything — it quietly
produces flattering numbers that make a losing strategy look profitable, and
you find out with real money. The look-ahead tests are the important ones.
"""

import unittest

import numpy as np
import pandas as pd

import costs
import data
import evaluate
import signals
import universe


def uptrend(n=70, start=100.0, step=0.5, vol=1e6):
    """A clean rising series, so the trend filter passes and patterns can be appended."""
    close = np.array([start + step * k for k in range(n)])
    open_ = close - step * 0.4
    high = close + step * 0.3
    low = open_ - step * 0.3
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": np.full(n, vol)},
                        index=pd.bdate_range("2020-01-01", periods=n))


def rally_pause_dip(pause_len, drop, step=2.0, amp=0.1, ndrops=4):
    """Rally, quiet sideways pause, sharp dip, then one up day."""
    rally = [100 + step * k for k in range(61)]
    pause = [rally[-1] + (amp if k % 2 else -amp) for k in range(pause_len)]
    dip = [pause[-1] - drop * (j + 1) for j in range(ndrops)]
    close = np.array(rally + pause + dip + [dip[-1] + drop * 1.2], float)
    prev = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"Open": prev, "Close": close},
                      index=pd.bdate_range("2020-01-01", periods=len(close)))
    df["High"] = df[["Open", "Close"]].max(axis=1) + 0.2
    df["Low"] = df[["Open", "Close"]].min(axis=1) - 0.2
    df["Volume"] = 1e6
    return df


def shaped(close, spike_at=()):
    """Bars built from a close path, with volume spikes on chosen bars."""
    close = np.asarray(close, float)
    d = np.diff(close, prepend=close[0])
    open_ = close - d * 0.4
    vol = np.full(len(close), 1e6)
    vol[list(spike_at)] = 5e6
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + 0.15,
                         "Low": np.minimum(open_, close) - 0.15, "Close": close,
                         "Volume": vol}, index=pd.bdate_range("2020-01-01", periods=len(close)))


def set_bar(df, i, o, h, l, c, v=None):
    df.iloc[i, df.columns.get_loc("Open")] = o
    df.iloc[i, df.columns.get_loc("High")] = h
    df.iloc[i, df.columns.get_loc("Low")] = l
    df.iloc[i, df.columns.get_loc("Close")] = c
    if v is not None:
        df.iloc[i, df.columns.get_loc("Volume")] = v


class TestIndicators(unittest.TestCase):
    def test_rsi_bounds(self):
        rising = pd.Series(np.arange(1, 40, dtype=float))
        self.assertAlmostEqual(signals.rsi(rising).iloc[-1], 100.0)
        self.assertAlmostEqual(signals.rsi(rising[::-1].reset_index(drop=True)).iloc[-1], 0.0)

    def test_rsi_midrange_for_choppy_prices(self):
        chop = pd.Series([100 + (1 if k % 2 else -1) for k in range(60)], dtype=float)
        self.assertTrue(30 < signals.rsi(chop).iloc[-1] < 70)

    def test_sma_matches_hand_calculation(self):
        df = uptrend(60)
        out = signals.add_indicators(df)
        expected = df["Close"].iloc[-20:].mean()
        self.assertAlmostEqual(out["pull_ma"].iloc[-1], expected, places=9)

    def test_atr_equals_range_for_flat_gaps(self):
        n = 40
        df = pd.DataFrame({"Open": 100.0, "High": 102.0, "Low": 98.0, "Close": 100.0,
                           "Volume": 1e6}, index=pd.bdate_range("2020-01-01", periods=n))
        self.assertAlmostEqual(signals.atr(df).iloc[-1], 4.0, places=6)


class TestPatternsFire(unittest.TestCase):
    """Textbook patterns must be detected."""

    def test_bullish_engulfing(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i - 1, o=132.0, h=132.5, l=129.5, c=130.0)     # red
        set_bar(df, i, o=129.5, h=136.0, l=129.0, c=135.5)         # green, engulfs it
        sig = signals.engulfing(signals.add_indicators(df), i)
        self.assertIsNotNone(sig)
        self.assertAlmostEqual(sig["trigger"], 136.0)
        self.assertAlmostEqual(sig["stop"], 129.0)
        self.assertAlmostEqual(sig["target"], 136.0 + 2 * 7.0)

    def test_hammer(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i, o=130.0, h=130.6, l=125.0, c=130.4)         # long lower wick
        self.assertIsNotNone(signals.hammer(signals.add_indicators(df), i))

    def test_piercing(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i - 1, o=134.0, h=134.5, l=128.0, c=128.5)     # big red
        set_bar(df, i, o=127.5, h=132.5, l=127.0, c=132.0)         # opens lower, closes past midpoint
        self.assertIsNotNone(signals.piercing(signals.add_indicators(df), i))

    def test_morning_star(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i - 2, o=134.0, h=134.5, l=127.5, c=128.0)     # big red
        set_bar(df, i - 1, o=128.0, h=129.0, l=127.0, c=128.3)     # small indecision
        set_bar(df, i, o=128.5, h=134.0, l=128.0, c=133.5)         # strong green
        self.assertIsNotNone(signals.morning_star(signals.add_indicators(df), i))

    def test_breakout_on_volume(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i, o=134.0, h=139.0, l=133.5, c=138.5, v=5e6)  # new high, 5x volume
        self.assertIsNotNone(signals.breakout(signals.add_indicators(df), i))

    def test_pullback(self):
        df = uptrend(n=80)
        i = len(df) - 1
        base = signals.add_indicators(df)["pull_ma"].iat[i]
        set_bar(df, i - 1, o=base + 1.0, h=base + 1.2, l=base - 0.5, c=base - 0.3)
        set_bar(df, i, o=base - 0.2, h=base + 3.0, l=base - 0.6, c=base + 2.5)
        self.assertIsNotNone(signals.pullback(signals.add_indicators(df), i))

    def test_rsi_bounce(self):
        # RSI under 35 while still above the 50-day average only happens in a narrow
        # shape: strong rally, a quiet pause, then a sharp shallow dip. A long pause
        # lets the average catch up; a smooth trend keeps RSI too high. See patterns.md.
        ind = signals.add_indicators(rally_pause_dip(pause_len=25, drop=2.0))
        i = len(ind) - 1
        self.assertLess(ind["rsi"].iloc[i - 4:i + 1].min(), signals.RSI_OVERSOLD)
        self.assertGreater(ind["Close"].iat[i], ind["trend_ma"].iat[i])
        self.assertIsNotNone(signals.rsi_bounce(ind, i))


class TestPatternsDoNotFire(unittest.TestCase):
    """Near misses must be rejected, or the scanner finds a 'setup' every day."""

    def test_engulfing_rejected_when_body_too_small(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i - 1, o=132.0, h=132.5, l=129.5, c=130.0)
        set_bar(df, i, o=129.9, h=132.2, l=129.8, c=132.1)         # barely bigger body
        self.assertIsNone(signals.engulfing(signals.add_indicators(df), i))

    def test_engulfing_rejected_in_downtrend(self):
        df = uptrend()
        df["Close"] = df["Close"].iloc[::-1].values                # now below its 50-day average
        df["Open"], df["High"], df["Low"] = df["Close"] + 0.2, df["Close"] + 0.5, df["Close"] - 0.5
        i = len(df) - 1
        set_bar(df, i - 1, o=102.0, h=102.5, l=99.5, c=100.0)
        set_bar(df, i, o=99.5, h=106.0, l=99.0, c=105.5)
        self.assertIsNone(signals.engulfing(signals.add_indicators(df), i))

    def test_pullback_rejected_when_price_never_dips_to_the_average(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i, o=133.0, h=136.0, l=132.8, c=135.5)         # wide green bar, far above the 20-day line
        ind = signals.add_indicators(df)
        self.assertGreater(ind["Low"].iloc[i - 2:i + 1].min(), ind["pull_ma"].iat[i] * signals.PULLBACK_TOUCH)
        # prove the bar would otherwise be acceptable, so the test cannot pass by accident
        self.assertIsNotNone(signals._package("pullback", 136.0, 132.8))
        self.assertIsNone(signals.pullback(ind, i))

    def test_pullback_rejected_on_a_red_candle(self):
        df = uptrend(n=80)
        i = len(df) - 1
        base = signals.add_indicators(df)["pull_ma"].iat[i]
        set_bar(df, i - 1, o=base + 1.0, h=base + 1.2, l=base - 0.5, c=base - 0.3)
        set_bar(df, i, o=base + 2.5, h=base + 3.0, l=base - 0.6, c=base - 0.2)   # red: still falling
        self.assertIsNone(signals.pullback(signals.add_indicators(df), i))

    def test_piercing_rejected_when_close_stays_below_midpoint(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i - 1, o=134.0, h=134.5, l=128.0, c=128.5)     # big red, midpoint 131.25
        set_bar(df, i, o=127.5, h=131.0, l=127.0, c=130.5)         # recovers, but not past halfway
        self.assertIsNone(signals.piercing(signals.add_indicators(df), i))

    def test_morning_star_rejected_when_middle_candle_is_large(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i - 2, o=134.0, h=134.5, l=127.5, c=128.0)
        set_bar(df, i - 1, o=128.0, h=131.5, l=127.0, c=131.0)     # not a small hesitant candle
        set_bar(df, i, o=128.5, h=134.0, l=128.0, c=133.5)
        self.assertIsNone(signals.morning_star(signals.add_indicators(df), i))

    def test_breakout_rejected_without_volume(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i, o=134.0, h=139.0, l=133.5, c=138.5)         # new high, ordinary volume
        self.assertIsNone(signals.breakout(signals.add_indicators(df), i))

    def test_rsi_bounce_rejected_when_dip_breaks_the_trend(self):
        # a long pause lets the 50-day average catch up, so the dip lands below it
        ind = signals.add_indicators(rally_pause_dip(pause_len=60, drop=1.0))
        i = len(ind) - 1
        self.assertLess(ind["rsi"].iloc[i - 4:i + 1].min(), signals.RSI_OVERSOLD)
        self.assertLess(ind["Close"].iat[i], ind["trend_ma"].iat[i])
        self.assertIsNone(signals.rsi_bounce(ind, i))

    def test_hammer_rejected_when_wick_is_short(self):
        df = uptrend()
        i = len(df) - 1
        set_bar(df, i, o=130.0, h=130.6, l=129.4, c=130.4)
        self.assertIsNone(signals.hammer(signals.add_indicators(df), i))

    def test_absurd_stop_distance_rejected(self):
        self.assertIsNone(signals._package("x", 100.0, 99.9))      # 0.1% risk: too tight
        self.assertIsNone(signals._package("x", 100.0, 80.0))      # 20% risk: too wide
        self.assertIsNone(signals._package("x", 100.0, 101.0))     # stop above entry


class TestNoLookAhead(unittest.TestCase):
    """
    The critical property: a signal on day T must not change when future days are
    removed. If it does, the backtest is reading tomorrow's prices and every
    result it prints is fiction.
    """

    def test_signals_identical_when_future_is_truncated(self):
        frames = data.demo_frames(n_symbols=4, years=4, seed=3)
        checked = 0
        for df in frames.values():
            full = signals.add_indicators(df)
            for i in range(len(df) - 40, len(df) - 5):
                truncated = signals.add_indicators(df.iloc[:i + 1])
                self.assertEqual(signals.detect_all(full, i),
                                 signals.detect_all(truncated, i),
                                 f"signal at bar {i} changed when future bars were removed")
                checked += 1
        self.assertGreater(checked, 100)

    def test_indicators_do_not_use_future_bars(self):
        df = data.demo_frames(n_symbols=1, years=3, seed=11)["DEMO1"]
        full = signals.add_indicators(df)
        cut = signals.add_indicators(df.iloc[:-10])
        for col in ("trend_ma", "pull_ma", "rsi", "atr", "vol_avg", "close_high"):
            self.assertAlmostEqual(full[col].iloc[-11], cut[col].iloc[-1], places=9, msg=col)


class TestTradeSimulation(unittest.TestCase):
    def _frame(self, bars):
        idx = pd.bdate_range("2021-01-01", periods=len(bars))
        return pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"], index=idx)

    SIG = {"signal": "t", "trigger": 100.0, "stop": 95.0, "target": 110.0}

    def test_no_entry_when_trigger_never_reached(self):
        df = self._frame([[99, 99.5, 98, 99], [98, 99.9, 97, 98]])
        self.assertIsNone(evaluate.simulate_trade(df, 0, self.SIG))

    def test_target_hit_is_a_win(self):
        df = self._frame([[99, 99.5, 98, 99], [100, 111, 99.5, 110.5]])
        t = evaluate.simulate_trade(df, 0, self.SIG)
        self.assertEqual(t["outcome"], "win")
        self.assertAlmostEqual(t["entry"], 100.0)
        self.assertAlmostEqual(t["exit"], 110.0)
        self.assertAlmostEqual(t["r_multiple"], 2.0)

    def test_stop_hit_is_a_loss(self):
        df = self._frame([[99, 99.5, 98, 99], [100, 101, 94, 94.5]])
        t = evaluate.simulate_trade(df, 0, self.SIG)
        self.assertEqual(t["outcome"], "loss")
        self.assertAlmostEqual(t["r_multiple"], -1.0)

    def test_both_hit_in_one_bar_counts_as_loss(self):
        """Daily bars cannot say which came first, so assume the worse one."""
        df = self._frame([[99, 99.5, 98, 99], [100, 112, 94, 105]])
        self.assertEqual(evaluate.simulate_trade(df, 0, self.SIG)["outcome"], "loss")

    def test_gap_above_target_is_skipped(self):
        df = self._frame([[99, 99.5, 98, 99], [115, 118, 114, 117]])
        self.assertIsNone(evaluate.simulate_trade(df, 0, self.SIG))

    def test_entry_uses_the_open_when_it_gaps_past_the_trigger(self):
        df = self._frame([[99, 99.5, 98, 99], [104, 111, 103, 110.5]])
        t = evaluate.simulate_trade(df, 0, self.SIG)
        self.assertAlmostEqual(t["entry"], 104.0)       # not the 100 trigger

    def test_unresolved_trade_closes_out_at_max_hold(self):
        bars = [[99, 99.5, 98, 99]] + [[100, 101, 99, 100]] * 20
        t = evaluate.simulate_trade(self._frame(bars), 0, self.SIG, max_hold=5)
        self.assertEqual(t["outcome"], "timeout")
        self.assertEqual(t["held_days"], 5)

    def test_charges_are_deducted(self):
        df = self._frame([[99, 99.5, 98, 99], [100, 111, 99.5, 110.5]])
        t = evaluate.simulate_trade(df, 0, self.SIG, notional=10000)
        gross = (t["exit"] - t["entry"]) * t["qty"]
        self.assertLess(t["net_rupees"], gross)
        self.assertAlmostEqual(t["net_rupees"],
                               gross - costs.round_trip_cost(t["entry"] * t["qty"],
                                                             t["exit"] * t["qty"]), places=6)


class TestEvaluatorEndToEnd(unittest.TestCase):
    def test_steady_rise_produces_only_wins(self):
        """When price only goes up, no triggered trade can ever reach its stop."""
        up = shaped([100 + 0.5 * k for k in range(150)], spike_at=range(60, 150, 10))
        trades = evaluate.collect_trades({"UP": signals.add_indicators(up)})
        self.assertGreater(len(trades), 0, "breakouts should fire on the volume spikes")
        self.assertTrue((trades["outcome"] == "win").all())

    def test_untriggered_breakout_before_a_crash_is_never_bought(self):
        """The buy-above-trigger rule must keep you out when price turns straight down."""
        close = [100 + 0.5 * k for k in range(80)] + [139.5 - 1.5 * k for k in range(1, 40)]
        ind = signals.add_indicators(shaped(close, spike_at=[79]))
        self.assertIsNotNone(signals.breakout(ind, 79), "setup should fire at the peak")
        self.assertTrue(evaluate.collect_trades({"CRASH": ind}).empty)

    def test_triggered_breakout_that_collapses_is_a_full_loss(self):
        close = [100 + 0.5 * k for k in range(80)] + [140.0] + [139.0 - 2.0 * k for k in range(1, 30)]
        df = shaped(close, spike_at=[79])
        set_bar(df, 80, o=139.7, h=141.0, l=139.5, c=140.0)
        ind = signals.add_indicators(df)
        sig = signals.breakout(ind, 79)
        trade = evaluate.simulate_trade(ind, 79, sig)
        self.assertEqual(trade["outcome"], "loss")
        self.assertAlmostEqual(trade["entry"], 139.7)          # gapped past the 139.65 trigger
        self.assertAlmostEqual(trade["r_multiple"], -1.0)

    def test_summary_fields_are_consistent(self):
        frames = data.load_frames(demo=True, years=5)
        trades = evaluate.collect_trades(frames)
        stats = evaluate.summarize(trades)
        self.assertTrue(stats)
        for name, s in stats.items():
            self.assertEqual(s["reliable"], s["samples"] >= evaluate.MIN_SAMPLE, name)
            self.assertTrue(0 <= s["win_rate"] <= 1, name)
            subset = trades[trades["signal"] == name]
            self.assertAlmostEqual(s["expectancy_rupees"], subset["net_rupees"].mean(), places=6)


class TestUniverseAndCache(unittest.TestCase):
    def test_liquidity_filter_rejects_thin_stocks(self):
        thin = signals.add_indicators(uptrend(n=300, vol=100))     # ~Rs 12k traded a day
        liquid = signals.add_indicators(uptrend(n=300, vol=5e6))
        self.assertFalse(universe.passes_filters(thin))
        self.assertTrue(universe.passes_filters(liquid))

    def test_short_history_rejected(self):
        self.assertFalse(universe.passes_filters(signals.add_indicators(uptrend(n=100, vol=5e6))))

    def test_fallback_symbol_list_is_usable(self):
        syms = universe.load_symbols()
        self.assertGreater(len(syms), 20)
        self.assertTrue(all(s.isascii() and s.strip() == s for s in syms))

    def test_cache_round_trip_with_mocked_download(self):
        """Simulates a yfinance download so the caching path is covered offline."""
        import sys
        import types
        import tempfile
        import pathlib

        frames = data.demo_frames(n_symbols=2, years=2, seed=9)
        cols = pd.MultiIndex.from_product([["AAA.NS", "BBB.NS"],
                                           ["Open", "High", "Low", "Close", "Volume"]])
        a, b = frames["DEMO1"], frames["DEMO2"]
        raw = pd.concat([a, b], axis=1)
        raw.columns = cols

        fake = types.ModuleType("yfinance")
        fake.download = lambda *args, **kwargs: raw
        saved = sys.modules.get("yfinance")
        sys.modules["yfinance"] = fake
        original_cache = data.CACHE
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data.CACHE = pathlib.Path(tmp)
                ok, failed = data.update_cache(["AAA", "BBB"], years=2, pause=0)
                self.assertEqual((ok, failed), (2, 0))
                out = data.load_frames(["AAA", "BBB"], demo=False)
                self.assertEqual(set(out), {"AAA", "BBB"})
                self.assertIn("rsi", out["AAA"].columns)
                # re-running must not duplicate rows
                data.update_cache(["AAA"], years=2, pause=0)
                self.assertEqual(len(data.read_cached("AAA")), len(a))
        finally:
            data.CACHE = original_cache
            if saved is not None:
                sys.modules["yfinance"] = saved
            else:
                del sys.modules["yfinance"]


class TestReviewFixes(unittest.TestCase):
    """One test per bug found in the line-by-line review, so none can come back."""

    def test_nse_list_excludes_restricted_series(self):
        """Bug 1: NSE's CSV has ' SERIES' with a leading space; BE/BZ must be dropped."""
        import sys
        import types
        import tempfile
        import pathlib
        csv = ("SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING\n"
               "GOODCO,Good Co,EQ,01-JAN-2000\n"
               "BADCO,Bad Co,BE,01-JAN-2000\n"
               "ZEDCO,Zed Co,BZ,01-JAN-2000\n")
        fake = types.ModuleType("requests")
        fake.get = lambda *a, **k: types.SimpleNamespace(text=csv, raise_for_status=lambda: None)
        saved, saved_file = sys.modules.get("requests"), universe.SYMBOLS_FILE
        sys.modules["requests"] = fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                universe.SYMBOLS_FILE = pathlib.Path(tmp) / "symbols.csv"
                self.assertEqual(universe.load_symbols(refresh=True), ["GOODCO"])
        finally:
            universe.SYMBOLS_FILE = saved_file
            if saved is not None:
                sys.modules["requests"] = saved
            else:
                del sys.modules["requests"]

    def test_cache_is_replaced_not_stitched(self):
        """Bug 2: after a split re-adjusts old prices, the cache must hold only the new series."""
        import sys
        import types
        import tempfile
        import pathlib
        base = data.demo_frames(n_symbols=1, years=1, seed=4)["DEMO1"]
        adjusted = base * 0.5                                  # a 2:1 split rewrites history
        adjusted["Volume"] = base["Volume"] * 2
        cols = pd.MultiIndex.from_product([["AAA.NS"], ["Open", "High", "Low", "Close", "Volume"]])
        fake = types.ModuleType("yfinance")
        saved, original = sys.modules.get("yfinance"), data.CACHE
        sys.modules["yfinance"] = fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data.CACHE = pathlib.Path(tmp)
                for frame in (base, adjusted):
                    raw = frame.copy()
                    raw.columns = cols
                    fake.download = lambda *a, _r=raw, **k: _r
                    data.update_cache(["AAA"], years=1, pause=0)
                cached = data.read_cached("AAA")
                self.assertEqual(len(cached), len(adjusted))
                self.assertTrue(np.allclose(cached["Close"].values, adjusted["Close"].values))
        finally:
            data.CACHE = original
            if saved is not None:
                sys.modules["yfinance"] = saved
            else:
                del sys.modules["yfinance"]

    def test_stale_stocks_are_dropped(self):
        """Bug 3: a suspended stock's months-old signal must never be shown as today's."""
        import scan
        live = uptrend(n=300)
        suspended = uptrend(n=240)                             # data stops 60 bars earlier
        kept, stale, latest = scan.fresh_frames({"LIVE": live, "GONE": suspended})
        self.assertEqual(set(kept), {"LIVE"})
        self.assertEqual(stale, 1)
        self.assertEqual(latest, live.index[-1])

    def test_repeated_signal_does_not_open_overlapping_trades(self):
        """Bug 4: a setup firing on consecutive days is one trade, not several."""
        close = [100 + 0.5 * k for k in range(150)]
        ind = signals.add_indicators(shaped(close, spike_at=range(60, 150)))  # fires every bar
        trades = evaluate.collect_trades({"UP": ind})
        self.assertGreater(len(trades), 0)
        entries = trades.sort_values("entry_date")
        # every trade must start after the previous one finished
        for prev, nxt in zip(entries.itertuples(), entries.iloc[1:].itertuples()):
            self.assertGreater(nxt.entry_date, prev.exit_date)

    def test_trade_cut_off_by_end_of_data_is_excluded(self):
        """Bug 5: a trade with no result yet is not a result."""
        idx = pd.bdate_range("2021-01-01", periods=4)
        df = pd.DataFrame([[99, 99.5, 98, 99], [100, 101, 99, 100.5],
                           [100.5, 101, 99, 100], [100, 101, 99, 100.2]],
                          columns=["Open", "High", "Low", "Close"], index=idx)
        sig = {"signal": "t", "trigger": 100.0, "stop": 95.0, "target": 110.0}
        self.assertEqual(evaluate.simulate_trade(df, 0, sig, max_hold=15)["outcome"], "open")

    def test_losing_and_thin_rules_are_not_proven(self):
        """Bug 6: only reliable, profitable rules may be recommended."""
        import scan
        stats = {"good": {"reliable": True, "expectancy_rupees": 12.0},
                 "loser": {"reliable": True, "expectancy_rupees": -8.0},
                 "thin": {"reliable": False, "expectancy_rupees": 50.0}}
        self.assertTrue(scan.proven("good", stats))
        self.assertFalse(scan.proven("loser", stats))
        self.assertFalse(scan.proven("thin", stats))
        self.assertFalse(scan.proven("missing", stats))

    def test_rupee_figures_state_their_trade_size(self):
        """Bug 7 and 9: say what trade size the rupees assume, and honour --min-turnover."""
        import scan
        df = signals.add_indicators(uptrend(n=300, vol=5e5))  # about Rs 1 cr/day
        sig = [{"signal": "pullback", "trigger": 250.0, "stop": 245.0, "target": 260.0}]
        stats = {"pullback": {"reliable": True, "samples": 300, "win_rate": 0.45,
                              "avg_win_rupees": 900.0, "avg_loss_rupees": -500.0,
                              "expectancy_rupees": 130.0, "expectancy_r": 0.2}}
        text = scan.describe("X", df, sig, stats, 10000, 2000, 0.01, min_turnover=0.5)
        self.assertIn("per Rs 10,000 trade", text)
        self.assertIn("fine", text)
        strict = scan.describe("X", df, sig, stats, 10000, 2000, 0.01, min_turnover=50)
        self.assertIn("THIN", strict)

    def test_symbol_input_is_normalised(self):
        import scan
        for raw in ("reliance", " RELIANCE.NS ", "Reliance.ns", "RELIANCE"):
            self.assertEqual(scan.clean_symbol(raw), "RELIANCE")

    def test_single_stock_analysis_uses_the_requested_stock(self):
        """Found by reading output: --symbol DEMO3 used to analyse DEMO1."""
        import io
        import sys
        import contextlib
        import scan
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = ["scan.py", "--demo", "--symbol", "demo3"]
        try:
            with contextlib.redirect_stdout(buf):
                scan.main()
        finally:
            sys.argv = argv
        self.assertIn("DEMO3 on", buf.getvalue())
        self.assertNotIn("DEMO1 on", buf.getvalue())

    def test_warns_when_charges_swallow_the_risk(self):
        """Found in the first real run: Rs 16 of charges against Rs 13 at risk."""
        import scan
        df = signals.add_indicators(uptrend(n=300, vol=5e6))
        sig = [{"signal": "pullback", "trigger": 229.0, "stop": 216.38, "target": 254.24}]
        small = scan.describe("X", df, sig, {}, None, 2000, 0.01, min_turnover=5)
        self.assertIn("WARNING: charges are", small)
        big = scan.describe("X", df, sig, {}, None, 500000, 0.01, min_turnover=5)
        self.assertNotIn("WARNING: charges are", big)

    def test_no_order_instructions_without_measured_rules(self):
        """Found in the first real run: unmeasured picks came with Kite instructions."""
        import io
        import sys
        import contextlib
        import scan
        saved = scan.STATS_FILE
        buf = io.StringIO()
        argv = sys.argv
        try:
            scan.STATS_FILE = scan.Path("/nonexistent/stats.json")
            sys.argv = ["scan.py", "--demo", "--include-unproven"]
            with contextlib.redirect_stdout(buf):
                scan.main()
        finally:
            scan.STATS_FILE, sys.argv = saved, argv
        out = buf.getvalue()
        self.assertIn("NOT RECOMMENDATIONS", out)
        self.assertNotIn("Place these yourself in Kite", out)

    def test_unfinished_candle_dropped_during_market_hours(self):
        df = uptrend(n=5)
        day = df.index[-1]
        during = pd.Timestamp(day.date()).tz_localize(data.MARKET_TZ) + pd.Timedelta(hours=13)
        after = pd.Timestamp(day.date()).tz_localize(data.MARKET_TZ) + pd.Timedelta(hours=18)
        next_day = during + pd.Timedelta(days=1)
        self.assertEqual(len(data.drop_unfinished_day(df, during)), 4)
        self.assertEqual(len(data.drop_unfinished_day(df, after)), 5)
        self.assertEqual(len(data.drop_unfinished_day(df, next_day)), 5)

    def test_position_size_never_exceeds_cash_or_risk(self):
        import scan
        self.assertEqual(scan.position_size(100.0, 95.0, 2000, 0.01), 4)    # Rs 20 risk / Rs 5
        self.assertEqual(scan.position_size(1500.0, 1450.0, 2000, 0.01), 0)  # Rs 50 risk > Rs 20
        self.assertEqual(scan.position_size(100.0, 99.9, 2000, 0.5), 19)    # capped by cash


def trade_table(per_year, signal="rule", start_year=2018, noise=50.0, seed=1, per_day=1):
    """
    Synthetic trades with a known average per year. per_year: list of means, one per
    year from start_year. per_day > 1 puts that many identical trades on each date.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k, mean in enumerate(per_year):
        days = pd.bdate_range(f"{start_year + k}-01-02", periods=60)
        vals = mean + rng.normal(0, noise, len(days))
        vals = vals - vals.mean() + mean          # make the year's mean exactly `mean`
        for d, v in zip(days, vals):
            for _ in range(per_day):
                rows.append({"signal": signal, "entry_date": d, "net_rupees": v,
                             "entry": 100.0, "exit": 100.0 + v / 100, "risk_per_share": 5.0})
    return pd.DataFrame(rows)


class TestStressTest(unittest.TestCase):
    def test_parallel_replay_matches_sequential(self):
        frames = data.load_frames(demo=True, years=4)
        one = evaluate.collect_trades(frames, workers=1)
        many = evaluate.collect_trades(frames, workers=3)
        self.assertTrue(one.equals(many))

    def test_split_and_year_figures(self):
        t = trade_table([10, 20, 30, 40, 50, 60, 70, 80])          # 2018..2025
        r = evaluate.stress_test(t, split="2023-01-01")["rule"]
        self.assertAlmostEqual(r["early"]["per_trade"], np.mean([10, 20, 30, 40, 50]), places=6)
        self.assertAlmostEqual(r["late"]["per_trade"], np.mean([60, 70, 80]), places=6)
        self.assertEqual(r["early"]["samples"], 300)
        self.assertEqual(r["positive_years"], 8)
        self.assertAlmostEqual(r["by_year"][2020]["per_trade"], 30, places=6)

    def test_clustering_ignores_duplicated_same_day_trades(self):
        """The reason clustering exists: 5 copies of one day's trade are one piece of evidence."""
        base = trade_table([15] * 8, noise=200)
        dup = trade_table([15] * 8, noise=200, per_day=5)
        t_naive_base = evaluate.naive_t(base["net_rupees"])
        t_naive_dup = evaluate.naive_t(dup["net_rupees"])
        t_clu_base = evaluate.clustered_t(base["net_rupees"], base["entry_date"])
        t_clu_dup = evaluate.clustered_t(dup["net_rupees"], dup["entry_date"])
        self.assertAlmostEqual(t_naive_dup / t_naive_base, np.sqrt(5), delta=0.05)
        self.assertAlmostEqual(t_clu_dup, t_clu_base, places=6)
        self.assertAlmostEqual(t_clu_base, t_naive_base, delta=0.05 * abs(t_naive_base))

    def test_pricing_at_your_size(self):
        t = pd.DataFrame({"entry": [100.0, 100.0], "exit": [110.0, 110.0],
                          "risk_per_share": [10.0, 25.0]})
        qty, net = evaluate.price_at_size(t, capital=2000, risk_pct=0.01)
        self.assertEqual(list(qty), [2, 0])                   # Rs 20 risk / Rs 10 = 2; / Rs 25 = 0
        self.assertAlmostEqual(net.iloc[0], 20 - costs.round_trip_cost(200.0, 220.0), places=6)
        self.assertTrue(np.isnan(net.iloc[1]))                # unaffordable: no result, not zero

    def test_steady_edge_holds_up(self):
        r = evaluate.stress_test(trade_table([20] * 8))["rule"]
        self.assertTrue(r["robust"], r["reasons"])
        self.assertEqual(r["reasons"], [])

    def test_edge_that_vanished_recently_is_fragile(self):
        r = evaluate.stress_test(trade_table([40, 40, 40, 40, 40, -10, -10, -10]))["rule"]
        self.assertFalse(r["robust"])
        self.assertTrue(any("from 2023" in x for x in r["reasons"]), r["reasons"])

    def test_edge_that_only_appeared_recently_is_fragile(self):
        r = evaluate.stress_test(trade_table([-10, -10, -10, -10, -10, 60, 60, 60]))["rule"]
        self.assertFalse(r["robust"])
        self.assertTrue(any("before 2023" in x for x in r["reasons"]), r["reasons"])

    def test_edge_from_a_few_lucky_years_is_fragile(self):
        r = evaluate.stress_test(trade_table([200, -10, -10, -10, 200, -10, -10, -10]))["rule"]
        self.assertFalse(r["robust"])
        self.assertTrue(any("only 2 of 8 years" in x for x in r["reasons"]), r["reasons"])

    def test_edge_smaller_than_noise_is_fragile(self):
        r = evaluate.stress_test(trade_table([3] * 8, noise=400))["rule"]
        self.assertFalse(r["robust"])
        self.assertTrue(any("could be luck" in x for x in r["reasons"]), r["reasons"])

    def test_losing_and_tiny_rules_are_not_stress_tested(self):
        losing = trade_table([-5] * 8, signal="loser")
        tiny = trade_table([50], signal="tiny").head(30)
        self.assertEqual(evaluate.stress_test(pd.concat([losing, tiny])), {})

    def test_saved_trades_reproduce_the_same_stats(self):
        import tempfile
        import pathlib
        frames = data.load_frames(demo=True, years=4)
        trades = evaluate.collect_trades(frames)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "trades.csv"
            trades.to_csv(path, index=False)
            reloaded, notional = evaluate.load_trades(path)
        self.assertEqual(notional, 10000.0)
        a, b = evaluate.summarize(trades), evaluate.summarize(reloaded)
        self.assertEqual(a.keys(), b.keys())
        for rule in a:                       # CSV round-trips floats to ~15 digits, not exactly
            for field, value in a[rule].items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, b[rule][field], places=9, msg=f"{rule}.{field}")
                else:
                    self.assertEqual(value, b[rule][field], msg=f"{rule}.{field}")

    def test_verdict_says_when_a_solid_rule_loses_at_your_size(self):
        import io
        import contextlib
        t = trade_table([20] * 8)
        t["notional"] = 10000.0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            evaluate.print_stress(evaluate.stress_test(t, capital=2000), "2023-01-01")
        self.assertIn("HOLDS UP at Rs 10,000 per trade — but LOSES MONEY at your Rs 2,000",
                      buf.getvalue())

    def test_scanner_requires_passing_the_stress_test(self):
        import scan
        base = {"reliable": True, "expectancy_rupees": 17.0}
        self.assertTrue(scan.proven("r", {"r": dict(base)}))                   # old stats.json
        self.assertTrue(scan.proven("r", {"r": dict(base, robust=True)}))
        self.assertFalse(scan.proven("r", {"r": dict(base, robust=False)}))


def make_panel(closes, opens=None, volume=1e6, start="2018-01-01"):
    """Wide price tables from {symbol: list of closes}. Opens default to the closes."""
    n = len(next(iter(closes.values())))
    idx = pd.bdate_range(start, periods=n)
    close = pd.DataFrame({k: np.asarray(v, float) for k, v in closes.items()}, index=idx)
    open_ = close.copy() if opens is None else pd.DataFrame(
        {k: np.asarray(v, float) for k, v in opens.items()}, index=idx)
    vol = volume if isinstance(volume, pd.DataFrame) else pd.DataFrame(volume, index=idx,
                                                                         columns=close.columns)
    return {"open": open_, "close": close, "volume": vol}


def growth(rate, n, start=100.0):
    return start * (1 + rate) ** np.arange(n)


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        import portfolio
        import strategies
        self.pf, self.st = portfolio, strategies

    def test_rebalance_on_last_trading_day_of_month(self):
        idx = pd.bdate_range("2021-01-01", "2021-03-31")
        days = self.pf.rebalance_days(idx, "M")
        self.assertEqual([d.date().isoformat() for d in days],
                         ["2021-01-29", "2021-02-26", "2021-03-31"])

    def test_momentum_buys_the_strongest_stock(self):
        n = 400
        panel = make_panel({"SLOW": growth(0.0002, n), "FAST": growth(0.002, n),
                            "MID": growth(0.001, n), "NIFTYBEES": growth(0.0005, n)})
        r = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=100000, top_n=1,
                        min_turnover_cr=0.1)
        bought = set(r.trades.loc[r.trades["side"] == "BUY", "symbol"])
        self.assertEqual(bought, {"FAST"})

    def test_trades_happen_at_the_next_days_open(self):
        n = 400
        opens = {k: v * 1.01 for k, v in {"FAST": growth(0.002, n), "SLOW": growth(0.0, n),
                                          "NIFTYBEES": growth(0.0005, n)}.items()}
        panel = make_panel({"FAST": growth(0.002, n), "SLOW": growth(0.0, n),
                            "NIFTYBEES": growth(0.0005, n)}, opens=opens)
        r = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=100000, top_n=1,
                        min_turnover_cr=0.1)
        idx = panel["close"].index
        rebal = set(self.pf.rebalance_days(idx, "M"))
        for t in r.trades.itertuples():
            decision = idx[idx.get_loc(t.date) - 1]
            self.assertIn(decision, rebal, "traded on a day that did not follow a decision")
            self.assertAlmostEqual(t.price, panel["open"].loc[t.date, t.symbol])

    def test_race_starts_when_the_strategy_can_first_decide(self):
        """Fairness: no year in cash while the benchmark is already invested."""
        n = 400
        panel = make_panel({"FAST": growth(0.002, n), "NIFTYBEES": growth(0.0005, n)})
        strat = self.st.STRATEGIES["momentum"]
        r = self.pf.run(panel, strat, capital=100000, top_n=1, min_turnover_cr=0.1)
        idx = panel["close"].index
        self.assertGreaterEqual(idx.get_loc(r.equity.index[0]), strat.min_history)
        first_buy = r.trades.iloc[0]["date"]
        self.assertEqual(idx.get_loc(first_buy), idx.get_loc(r.equity.index[0]) + 1)

    def test_no_look_ahead_in_any_strategy(self):
        """A pick on day i must not change when every later day is deleted."""
        panel = data.load_panel(demo=True, years=3, n_demo=25)
        idx = panel["close"].index
        for key, strat in self.st.STRATEGIES.items():
            for i in (520, 600, 700):
                full_ctx = self.st.Context(panel["close"], panel["open"], panel["volume"])
                cut = {k: v.iloc[:i + 1] for k, v in panel.items()}
                cut_ctx = self.st.Context(cut["close"], cut["open"], cut["volume"])
                args = lambda c: (c.close.notna().cumsum(),
                                  (c.close * c.volume).rolling(60, min_periods=40).median())
                a = self.pf.pick(full_ctx, i, strat, 0.1, *args(full_ctx), strat.top_n)
                b = self.pf.pick(cut_ctx, i, strat, 0.1, *args(cut_ctx), strat.top_n)
                self.assertEqual(a, b, f"{key} at {idx[i].date()} used future prices")

    def test_round_trip_cash_is_exact(self):
        """Buy, then sell when the stock drops out: cash must match the arithmetic exactly."""
        n = 420
        a = np.r_[growth(0.002, 300), np.full(n - 300, growth(0.002, 300)[-1] * 0.7)]
        b = np.r_[growth(0.0, 300), growth(0.003, n - 300, start=100.0)]
        panel = make_panel({"A": a, "B": b, "NIFTYBEES": growth(0.0, n)})
        r = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=100000, top_n=1,
                        min_turnover_cr=0.1)
        cash = 100000.0
        for t in r.trades.itertuples():
            cash += -(t.value + t.charges) if t.side == "BUY" else (t.value - t.charges)
            expected = costs.trade_cost(t.value, "buy" if t.side == "BUY" else "sell")
            self.assertAlmostEqual(t.charges, expected, places=6)
        self.assertIn("SELL", set(r.trades["side"]))
        # after the final trade, equity = cash + value of whatever is still held
        still = {}
        for t in r.trades.itertuples():
            still[t.symbol] = still.get(t.symbol, 0) + (t.qty if t.side == "BUY" else -t.qty)
        value = sum(q * panel["close"][s].iloc[-1] for s, q in still.items() if q)
        self.assertAlmostEqual(r.equity.iloc[-1], cash + value, places=4)

    def test_small_capital_cannot_buy_expensive_stocks(self):
        n = 400
        panel = make_panel({"A": growth(0.002, n, 5000), "B": growth(0.001, n, 6000),
                            "NIFTYBEES": growth(0.0005, n, 250)})
        r = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=2000, top_n=2,
                        min_turnover_cr=0.1)
        self.assertEqual(len(r.trades), 0)
        self.assertGreater(r.metrics["unaffordable"], 0)
        self.assertTrue((r.equity == 2000).all())

    def test_never_puts_more_than_an_equal_share_into_one_stock(self):
        """Rs 10,000 split two ways is Rs 5,000 each: a Rs 6,000 share must be skipped."""
        n = 400
        panel = make_panel({"A": growth(0.002, n, 6000), "B": growth(0.0015, n, 6000),
                            "NIFTYBEES": growth(0.0005, n, 250)})
        r = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=10000, top_n=2,
                        min_turnover_cr=0.1)
        self.assertEqual(len(r.trades), 0)
        self.assertGreaterEqual(r.metrics["unaffordable"], 2)

    def test_stock_strategies_never_hold_the_benchmark(self):
        n = 400
        rng = np.random.default_rng(3)
        noisy = lambda: 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        panel = make_panel({"A": noisy(), "B": noisy(), "NIFTYBEES": growth(0.0003, n)})
        r = self.pf.run(panel, self.st.STRATEGIES["low_vol"], capital=100000, top_n=1,
                        min_turnover_cr=0.1)
        self.assertNotIn("NIFTYBEES", set(r.trades["symbol"]))
        self.assertGreater(len(r.trades), 0)

    def test_liquidity_is_judged_point_in_time(self):
        n = 600
        idx = pd.bdate_range("2018-01-01", periods=n)
        vol = pd.DataFrame(1e6, index=idx, columns=["THIN", "OK", "NIFTYBEES"])
        vol.loc[idx[:400], "THIN"] = 10                       # untradeable for its first 400 days
        panel = make_panel({"THIN": growth(0.003, n), "OK": growth(0.001, n),
                            "NIFTYBEES": growth(0.0005, n)}, volume=vol)
        r = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=100000, top_n=1,
                        min_turnover_cr=0.5)
        buys = r.trades[r.trades["side"] == "BUY"]
        first_thin = buys.loc[buys["symbol"] == "THIN", "date"].min()
        self.assertEqual(buys.iloc[0]["symbol"], "OK")
        self.assertGreater(first_thin, idx[400])              # bought only once it was liquid

    def test_index_trend_moves_to_cash_below_the_average(self):
        n = 520
        path = np.r_[growth(0.002, 400, 100), growth(-0.01, n - 400, growth(0.002, 400, 100)[-1])]
        panel = make_panel({"NIFTYBEES": path, "X": growth(0.0, n)})
        r = self.pf.run(panel, self.st.STRATEGIES["index_trend"], capital=100000,
                        min_turnover_cr=0.1)
        sides = list(r.trades["side"])
        self.assertEqual(sides[0], "BUY")
        self.assertEqual(sides[-1], "SELL")
        held_at_end = r.equity.iloc[-1] - r.equity.iloc[-2]
        self.assertAlmostEqual(held_at_end, 0.0, places=6)      # all cash: value stops moving
        nb = r.trades[r.trades["symbol"] == "NIFTYBEES"].iloc[0]
        self.assertAlmostEqual(nb["charges"], costs.trade_cost(nb["value"], "buy", etf=True))

    def test_benchmark_starts_with_the_strategy(self):
        n = 400
        panel = make_panel({"A": growth(0.002, n), "NIFTYBEES": growth(0.001, n, 250)})
        r = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=100000, top_n=1,
                        min_turnover_cr=0.1)
        self.assertEqual(r.benchmark.iloc[0], 100000)
        self.assertEqual(r.benchmark.index[0], r.equity.index[0])
        day1 = r.benchmark.index[1]
        price = panel["open"].loc[day1, "NIFTYBEES"]
        units = int(100000 * (1 - self.pf.BUY_BUFFER) // price)
        fee = costs.trade_cost(units * price, "buy", etf=True)
        expected = 100000 - units * price - fee + units * panel["close"].loc[day1, "NIFTYBEES"]
        self.assertAlmostEqual(r.benchmark.iloc[1], expected, places=6)

    def test_cagr_and_drawdown_known_answers(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-12-31"])
        days = (idx[1] - idx[0]).days
        s = pd.Series([100.0, 100 * 2 ** (days / 365.25)], index=idx)
        self.assertAlmostEqual(self.pf.cagr(s), 1.0, places=9)
        dd = pd.Series([100, 120, 60, 90.0], index=pd.bdate_range("2020-01-01", periods=4))
        self.assertAlmostEqual(self.pf.max_drawdown(dd), -0.5)

    def test_verdict_requires_a_steady_lead(self):
        idx = pd.bdate_range("2018-01-01", "2025-12-31")
        rng = np.random.default_rng(5)
        bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))), index=idx)
        steady = bench * np.exp(np.arange(len(idx)) * 0.0004)            # +~10%/yr, every month
        yearly = self.pf.yearly_returns(steady, bench)
        self.assertTrue(self.pf.judge(steady, bench, yearly)["beats_benchmark"])
        same = self.pf.judge(bench.copy(), bench, self.pf.yearly_returns(bench.copy(), bench))
        self.assertFalse(same["beats_benchmark"])
        self.assertTrue(any("could be luck" in x for x in same["reasons"]))

    def test_verdict_rejects_a_lead_from_a_few_big_years(self):
        """Passes on average and on confidence, but beat the index in only 2 of 6 full years."""
        idx = pd.bdate_range("2018-01-01", "2025-12-31")
        bench = pd.Series(100 * np.exp(np.arange(len(idx)) * 0.0003), index=idx)
        big = idx.year.isin([2018, 2020, 2023, 2025])
        excess = np.where(big, 0.002, -0.0001)
        strat = bench * np.exp(np.cumsum(excess))
        v = self.pf.judge(strat, bench, self.pf.yearly_returns(strat, bench))
        self.assertFalse(v["beats_benchmark"])
        self.assertEqual(len(v["reasons"]), 1, v["reasons"])
        self.assertIn("only 2 of 6 full years", v["reasons"][0])

    def test_mean_reversion_only_buys_dips_in_uptrends(self):
        n = 260
        up_then_dip = np.r_[growth(0.003, n - 5), growth(0.003, n - 5)[-1] * np.array(
            [0.98, 0.96, 0.95, 0.94, 0.93])]
        down_then_dip = np.r_[growth(-0.002, n - 5), growth(-0.002, n - 5)[-1] * np.array(
            [0.97, 0.94, 0.92, 0.90, 0.88])]
        panel = make_panel({"UP": up_then_dip, "DOWN": down_then_dip, "NIFTYBEES": growth(0, n)})
        ctx = self.st.Context(panel["close"], panel["open"], panel["volume"])
        scores = self.st.mean_reversion_score(ctx, n - 1)
        self.assertFalse(np.isnan(scores["UP"]))          # fell, but still above its 200-day average
        self.assertTrue(np.isnan(scores["DOWN"]))         # fell further, but in a downtrend

    def test_unsellable_day_retries_next_day(self):
        n = 420
        a = np.r_[growth(0.002, 300), np.full(n - 300, growth(0.002, 300)[-1] * 0.7)]
        b = np.r_[growth(0.0, 300), growth(0.003, n - 300)]
        panel = make_panel({"A": a, "B": b, "NIFTYBEES": growth(0.0, n)})
        clean = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=100000, top_n=1,
                            min_turnover_cr=0.1)
        sell = clean.trades[(clean.trades["symbol"] == "A") & (clean.trades["side"] == "SELL")].iloc[0]
        panel["open"].loc[sell["date"], "A"] = np.nan             # no trading in A that day
        r = self.pf.run(panel, self.st.STRATEGIES["momentum"], capital=100000, top_n=1,
                        min_turnover_cr=0.1)
        retry = r.trades[(r.trades["symbol"] == "A") & (r.trades["side"] == "SELL")].iloc[0]
        nxt = panel["close"].index[panel["close"].index.get_loc(sell["date"]) + 1]
        self.assertEqual(retry["date"], nxt)


class TestDashboardAndPanel(unittest.TestCase):
    def test_panel_loads_real_cache_with_benchmark_and_reuses_pickle(self):
        import tempfile
        import pathlib
        frames = data.demo_frames(n_symbols=3, years=2, seed=12)
        frames["NIFTYBEES"] = data.demo_benchmark(frames)
        original = (data.CACHE, data.PANEL_CACHE, universe.load_symbols)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data.CACHE = pathlib.Path(tmp)
                data.PANEL_CACHE = data.CACHE / "_panel.pkl"
                for sym, df in frames.items():
                    df.rename_axis("Date").to_csv(data.CACHE / f"{sym}.csv")
                universe.load_symbols = lambda refresh=False: ["DEMO1", "DEMO2", "DEMO3"]
                panel = data.load_panel()
                self.assertEqual(set(panel["close"].columns), set(frames))   # benchmark added
                self.assertTrue(data.PANEL_CACHE.exists())
                again = data.load_panel()                                     # from the pickle
                self.assertTrue(again["close"].equals(panel["close"]))
        finally:
            data.CACHE, data.PANEL_CACHE, universe.load_symbols = original

    @unittest.skipUnless(__import__("importlib").util.find_spec("streamlit"),
                         "streamlit not installed")
    def test_dashboard_runs_every_strategy_without_errors(self):
        import os
        from streamlit.testing.v1 import AppTest
        os.environ["LAB_FORCE_DEMO"] = "1"
        try:
            at = AppTest.from_file("app.py", default_timeout=180).run()
            self.assertEqual([e.value for e in at.exception], [])
            self.assertEqual([t.label for t in at.tabs],
                             ["Today's picks", "Strategy Lab", "Candlestick rules",
                              "Today's scan", "Limits"])
            for label in ["Low volatility", "Short-term mean reversion", "Index trend following"]:
                at.selectbox[0].set_value(label)
                at.button[0].click().run()
                self.assertEqual([e.value for e in at.exception], [], label)
                verdicts = [x.value for x in list(at.success) + list(at.error)]
                self.assertTrue(any("NIFTYBEES" in v for v in verdicts), label)
        finally:
            del os.environ["LAB_FORCE_DEMO"]

    def test_no_order_placing_code_anywhere(self):
        """The lab is research-only: no broker API, no order functions, anywhere."""
        import pathlib
        banned = ("kiteconnect", "place_order", "KiteConnect", "modify_order")
        for f in pathlib.Path(".").glob("*.py"):
            if f.name == "tests.py":
                continue
            text = f.read_text()
            for word in banned:
                self.assertNotIn(word, text, f"{f.name} contains {word}")


GOOGLE_RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Crude oil jumps 4% as Middle East conflict widens - Economic Times</title>
<link>https://example.com/a</link><pubDate>Fri, 02 Oct 2026 08:00:00 GMT</pubDate>
<source url="https://economictimes.com">Economic Times</source></item>
<item><title>RBI keeps repo rate unchanged - CNBC TV18</title>
<link>https://example.com/b</link><pubDate>Fri, 02 Oct 2026 06:00:00 GMT</pubDate>
<source url="https://cnbctv18.com">CNBC TV18</source></item>
<item><title>Company wins award for design - Mint</title>
<link>https://example.com/c</link><pubDate>Fri, 02 Oct 2026 05:00:00 GMT</pubDate>
<source url="https://livemint.com">Mint</source></item>
</channel></rss>"""

PUBLISHER_RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Sensex - Nifty end flat; IT stocks gain as rupee weakens</title>
<link>https://example.com/d</link><pubDate>Fri, 02 Oct 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


class TestNews(unittest.TestCase):
    def setUp(self):
        import news
        self.news = news

    def test_google_news_titles_lose_the_publisher_suffix(self):
        items = self.news.parse_feed(GOOGLE_RSS)
        self.assertEqual(items[0]["title"], "Crude oil jumps 4% as Middle East conflict widens")
        self.assertEqual(items[0]["source"], "Economic Times")
        self.assertEqual(str(items[0]["published"].tz), "Asia/Kolkata")

    def test_publisher_titles_keep_their_dashes(self):
        items = self.news.parse_feed(PUBLISHER_RSS, source="Mint Markets")
        self.assertEqual(items[0]["title"], "Sensex - Nifty end flat; IT stocks gain as rupee weakens")
        self.assertEqual(items[0]["source"], "Mint Markets")

    def test_broken_feed_gives_nothing_instead_of_crashing(self):
        self.assertEqual(self.news.parse_feed("<html>not rss"), [])

    def test_sector_tags(self):
        tags = lambda t: [s for s, _ in self.news.tag_sectors(t)]
        self.assertIn("Oil Gas & Consumable Fuels", tags("Crude oil jumps on OPEC cut"))
        self.assertIn(self.news.ALL, tags("Missile attack raises border tension"))
        self.assertIn("Financial Services", tags("RBI keeps repo rate unchanged"))
        self.assertIn("Information Technology", tags("IT stocks gain as rupee weakens"))
        self.assertEqual(tags("Company wins award for design"), [])      # 'war' inside 'award'
        self.assertEqual(tags("Bank shares rally"), [])                   # 'ban' inside 'bank'

    def test_red_flags(self):
        self.assertEqual(self.news.red_flags("SEBI bans promoter over fraud"),
                         ["regulator", "fraud or governance"])
        self.assertEqual(self.news.red_flags("TCS Q2 results: net profit rises"), ["results"])
        self.assertEqual(self.news.red_flags("Bank shares rally on strong demand"), [])

    def test_market_news_survives_failing_feeds_and_dedupes(self):
        def fake(url):
            if "economictimes" in url:
                raise ConnectionError("down")
            return GOOGLE_RSS if "news.google.com" in url else PUBLISHER_RSS
        df, failed = self.news.market_news(fetcher=fake, max_age_days=None)
        self.assertEqual(len(df), 4)                       # 3 unique Google items + 1 publisher
        self.assertTrue(any("Economic Times Markets" in f for f in failed))
        self.assertEqual(list(df["published"]), sorted(df["published"], reverse=True))

    def test_news_for_industry_includes_market_wide_items(self):
        df, _ = self.news.market_news(fetcher=lambda u: GOOGLE_RSS, max_age_days=None)
        it = self.news.news_for_industry(df, "Information Technology")
        self.assertIn("Crude oil jumps 4% as Middle East conflict widens", list(it["title"]))
        self.assertNotIn("Company wins award for design", list(it["title"]))

    def test_company_news_marks_flags(self):
        rss = GOOGLE_RSS.replace("RBI keeps repo rate unchanged", "SEBI probes XYZ Ltd")
        df, _ = self.news.company_news("XYZ", "XYZ Ltd", fetcher=lambda u: rss, max_age_days=None)
        self.assertEqual(df.loc[df["title"] == "SEBI probes XYZ Ltd", "flags"].iloc[0],
                         ["regulator"])


class TestPicks(unittest.TestCase):
    def setUp(self):
        import picks
        import portfolio
        self.picks, self.pf = picks, portfolio

    def test_minimum_position_keeps_charges_under_one_percent(self):
        floor = self.picks.min_position()
        self.assertLessEqual(costs.round_trip_cost(floor, floor) / floor, 0.01)
        self.assertGreater(costs.round_trip_cost(floor - 50, floor - 50) / (floor - 50), 0.01)

    def test_small_budget_goes_to_niftybees(self):
        prices = {"A": 100.0, "B": 200.0, "NIFTYBEES": 250.0}
        rows, cash, note = self.picks.plan_budget(2000, ["A", "B"], prices)
        self.assertEqual([r["symbol"] for r in rows], ["NIFTYBEES"])
        self.assertEqual(rows[0]["qty"], 7)
        self.assertIn("too small", note)
        self.assertAlmostEqual(rows[0]["buy_charges"],
                               costs.trade_cost(7 * 250.0, "buy", etf=True))

    def test_mid_budget_is_split_into_positions_that_fit(self):
        """Rs 10,000 with Rs 100 shares: 3 positions of ~Rs 3,300, not 5 that are too small."""
        prices = {f"S{i}": 100.0 for i in range(10)} | {"NIFTYBEES": 250.0}
        rows, cash, _ = self.picks.plan_budget(10000, [f"S{i}" for i in range(10)], prices)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["qty"] for r in rows], [33, 33, 33])
        self.assertLess(cash, 200)

    def test_awkwardly_priced_stock_is_skipped_not_bought_too_small(self):
        """Rs 6,000 -> 2 slots of Rs 3,000. One Rs 1,600 share is under the Rs 1,973 minimum."""
        prices = {"AWKWARD": 1600.0, "S1": 100.0, "S2": 100.0, "NIFTYBEES": 250.0}
        rows, _, _ = self.picks.plan_budget(6000, ["AWKWARD", "S1", "S2"], prices)
        self.assertEqual([r["symbol"] for r in rows], ["S1", "S2"])

    def test_no_passing_strategy_means_niftybees(self):
        rows, _, note = self.picks.plan_budget(100000, [], {"NIFTYBEES": 250.0})
        self.assertEqual(rows[0]["symbol"], "NIFTYBEES")
        self.assertIn("No strategy beat NIFTYBEES", note)

    def test_every_stock_position_clears_the_minimum_and_budget(self):
        rng = np.random.default_rng(8)
        floor = self.picks.min_position()
        for budget in (5000, 10000, 23000, 50000, 200000):
            ranked = [f"S{i}" for i in range(30)]
            prices = {s: float(rng.uniform(50, 6000)) for s in ranked}
            prices["NIFTYBEES"] = 250.0
            rows, cash, _ = self.picks.plan_budget(budget, ranked, prices)
            spent = sum(r["cost"] + r["buy_charges"] for r in rows)
            self.assertAlmostEqual(spent + cash, budget, places=6)
            self.assertLessEqual(len(rows), 5)
            for r in rows:
                if r["symbol"] != "NIFTYBEES":
                    self.assertGreaterEqual(r["cost"], floor, (budget, r))

    def test_strategy_choice_requires_passing_and_prefers_confidence(self):
        from types import SimpleNamespace as NS
        def res(beats, t):
            return NS(verdict={"beats_benchmark": beats, "t": t, "reasons": []},
                      metrics={"cagr": 0.1, "bench_cagr": 0.08})
        best, _ = self.picks.choose_strategy({"momentum": res(True, 2.1), "low_vol": res(True, 2.9),
                                              "mean_reversion": res(False, 5.0)})
        self.assertEqual(best, "low_vol")
        none, _ = self.picks.choose_strategy({"momentum": res(False, 1.0)})
        self.assertIsNone(none)

    def test_current_picks_match_the_engines_choice(self):
        n = 400
        panel = make_panel({"SLOW": growth(0.0002, n), "FAST": growth(0.002, n),
                            "MID": growth(0.001, n), "NIFTYBEES": growth(0.0005, n)})
        self.assertEqual(self.picks.current_picks(panel, "momentum", 0.1, n=2), ["FAST", "MID"])

    def test_hold_or_sell(self):
        held = pd.DataFrame({"symbol": ["fast.ns", "SLOW", "NIFTYBEES", "GONE", None],
                             "qty": [10, 5, 7, 3, None], "buy_price": [100, 100, 250, 50, None]})
        prices = {"FAST": 120.0, "SLOW": 90.0, "NIFTYBEES": 260.0}
        out = self.picks.review_holdings(held, prices, keep=["FAST"], key="momentum").set_index("symbol")
        self.assertTrue(out.loc["FAST", "action"].startswith("HOLD"))
        self.assertTrue(out.loc["SLOW", "action"].startswith("SELL"))
        self.assertTrue(out.loc["NIFTYBEES", "action"].startswith("HOLD"))
        self.assertIn("No price data", out.loc["GONE", "action"])
        self.assertAlmostEqual(out.loc["FAST", "profit_after_sell_charges"],
                               1200 - 1000 - costs.trade_cost(1200, "sell"))
        no_strategy = self.picks.review_holdings(held, prices, keep=None, key=None)
        self.assertIn("No tested strategy", no_strategy.set_index("symbol").loc["FAST", "action"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("streamlit"),
                         "streamlit not installed")
    def test_picks_tab_when_a_strategy_passes(self):
        """Force a pass so the picks, plan and news-free layout are exercised."""
        import os
        import picks
        from streamlit.testing.v1 import AppTest
        original = picks.choose_strategy
        def forced(results):
            _, rows = original(results)
            for r in rows:
                r["beats"] = r["key"] == "momentum"
            return "momentum", rows
        picks.choose_strategy = forced
        os.environ["LAB_FORCE_DEMO"] = "1"
        try:
            at = AppTest.from_file("app.py", default_timeout=240)
            at.run()
            self.assertEqual([e.value for e in at.exception], [])
            tab = at.tabs[0]
            self.assertTrue(any("Using **Momentum**" in x.value for x in tab.success))
            self.assertTrue(any(m.value.startswith("### Top 5") for m in tab.markdown))
        finally:
            picks.choose_strategy = original
            del os.environ["LAB_FORCE_DEMO"]


if __name__ == "__main__":
    unittest.main(verbosity=2)

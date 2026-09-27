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


if __name__ == "__main__":
    unittest.main(verbosity=2)

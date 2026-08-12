"""Data pipeline for the S&P 500 regime nowcast (spec section 3).

Fetches S&P 500 (and VIX) daily closes, resamples to a weekly observation
series (r_t = weekly log return, v_t = log realized vol from daily returns
within the week), and produces a train/test split.

Nothing in this module looks past the data available at each row's own
week -- the weekly resampling only aggregates *within* that week.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

SP500_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

# Weekly bars are anchored on Friday closes (calendar week), per spec section 3.
WEEKLY_ANCHOR = "W-FRI"

# MACRO channels (credit spread, yield-curve inversion) are loaded from static
# monthly FRED CSVs living next to this file, NOT fetched at runtime: the good
# free full-history series (Moody's BAA/AAA back to 1919, GS10/TB3MS back to the
# 1950s) are only reliably downloadable by hand, and one (HY OAS) was truncated to
# 3yr by FRED in Apr-2026. Committing the CSVs keeps the pipeline reproducible and
# offline. Column name in each CSV == the FRED series id.
_DATASET_DIR = pathlib.Path(__file__).resolve().parent
_MACRO_SERIES = {  # FRED id -> csv filename
    "BAA": "BAA.csv",       # Moody's Baa corporate yield (monthly, 1919+)
    "AAA": "AAA.csv",       # Moody's Aaa corporate yield (monthly, 1919+)
    "GS10": "GS10.csv",     # 10yr Treasury constant maturity (monthly, 1953+)
    "TB3MS": "TB3MS.csv",   # 3mo T-bill secondary market (monthly, 1934+)
}


def _load_macro_series(fred_id: str) -> pd.Series:
    """Read one monthly FRED CSV from the dataset dir as a date-indexed Series.

    CSVs are the raw FRED `observation_date,<ID>` download. Returned monthly (as
    published); callers resample to the weekly cadence CAUSALLY (see the feature
    methods -- forward-fill the last KNOWN monthly value, never interpolate forward).
    """
    path = _DATASET_DIR / _MACRO_SERIES[fred_id]
    df = pd.read_csv(path, parse_dates=["observation_date"]).set_index("observation_date")
    s = pd.to_numeric(df[fred_id], errors="coerce").dropna().sort_index()
    s.name = fred_id
    return s


def fetch_daily_closes(
    ticker: str,
    start: str = "1928-01-01",
    end: str | None = None,
) -> pd.Series:
    """Download daily close prices for `ticker` as a plain Series indexed by date."""
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}")
    closes = raw["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    closes.name = ticker
    return closes.dropna()


def weekly_log_returns(daily_close: pd.Series, anchor: str = WEEKLY_ANCHOR) -> pd.Series:
    """r_t: log return of the last daily close in each calendar week vs. the prior week."""
    weekly_close = daily_close.resample(anchor).last().dropna()
    log_price = np.log(weekly_close)
    r = log_price.diff().dropna()
    r.name = "r_t"
    return r


def weekly_log_realized_vol(daily_close: pd.Series, anchor: str = WEEKLY_ANCHOR) -> pd.Series:
    """v_t: log of realized vol, computed from daily log returns *within* each week only.

    Realized vol for week t uses only the daily closes belonging to week t, so this
    stays a same-week (not lookahead) aggregate, matching the weekly filtering cadence
    the model runs at.
    """
    daily_log_ret = np.log(daily_close).diff().dropna()
    weekly_groups = daily_log_ret.resample(anchor)
    # A week needs >=2 daily returns for realized vol to be defined; weeks with
    # exactly 1 (e.g. the 9/11/2001 week, when markets were shut for four days)
    # give a degenerate std of 0 -> log(0) = -inf, so they're dropped rather than
    # kept as a spurious "zero volatility" reading.
    counts = weekly_groups.count()
    realized_vol = weekly_groups.std(ddof=0)
    realized_vol = realized_vol[counts >= 2].dropna()
    log_rv = np.log(realized_vol)
    log_rv.name = "v_t"
    return log_rv


def weekly_log_vix(vix_daily_close: pd.Series, anchor: str = WEEKLY_ANCHOR) -> pd.Series:
    """Alternative v_t: log of the last VIX close observed in each week."""
    weekly_vix = vix_daily_close.resample(anchor).last().dropna()
    log_vix = np.log(weekly_vix)
    log_vix.name = "v_t_vix"
    return log_vix


@dataclass
class RegimeDataset:
    """Weekly observation series plus the daily price series used to build it."""

    daily_close: pd.Series  # S&P 500 daily closes, for the drawdown label (section 4)
    weekly_price: pd.Series  # last daily close per week, aligned to r_t / v_t index
    r_t: pd.Series  # weekly log return
    v_t: pd.Series  # weekly log realized vol (default v_t channel)
    v_t_vix: pd.Series | None  # weekly log VIX, if fetched (section 9 open question)
    # Raw MONTHLY macro series (loaded from static FRED CSVs); None if not loaded.
    # Kept monthly here -- the feature methods resample to weekly causally.
    macro: dict[str, pd.Series] | None = None

    def _monthly_to_weekly_causal(self, s: pd.Series) -> pd.Series:
        """Align a MONTHLY macro series onto the weekly r_t/v_t index CAUSALLY.

        Forward-fill only: each week gets the LAST monthly value published on or
        before it (FRED monthly obs are dated to the first of the month, i.e. the
        month they summarize). No interpolation, no lookahead -- a week in the
        middle of a month sees only the already-released prior month's figure, so
        this stays a same-or-past aggregate like every other channel.
        """
        combined = s.reindex(s.index.union(self.r_t.index)).sort_index().ffill()
        return combined.reindex(self.r_t.index)

    def credit_spread_change(self, horizon_months: int = 5) -> pd.Series:
        """cs_chg_t = (BAA - AAA) minus its value `horizon_months` ago -- the WIDENING
        MOMENTUM of the investment-grade quality spread. A CAUSAL, P&S-free macro feature.

        WHY THE CHANGE, NOT THE LEVEL: the raw BAA-AAA LEVEL is blind at a shallow bear
        onset (it sits in the normal bull band; measured onset separation ~-0.09sd). Its
        multi-MONTH CHANGE is what carries bear signal (spreads WIDEN as a default-driven
        bear develops). Separation on true bears: ~+0.5sd, and it is nearly
        UNCORRELATED with the drawdown / return / vol channels (adds new information rather
        than re-encoding price -- the bar the drawdown channel had to clear).
        HORIZON = 5 MONTHS: chosen from a 1-24 month sweep as the PEAK of a broad, smooth
        plateau (h=3..8 all give ~0.46-0.55sd; no lone spike), i.e. robust, not overfit to
        one window. min 3 / max 8 all defensible; 5 is the peak.
        ERA CAVEAT: this channel is strong in DEFAULT-driven bears (dotcom +0.95, GFC,
        post-2010 +1.05) but DEAD in the rate-driven 1970-82 stagflation bears (~-0.09) --
        there credit decoupled from equities and the yield-curve inversion channel covers
        instead (see curve_inversion). So the two macro channels are complementary by era.
        """
        cs = self.macro["BAA"] - self.macro["AAA"]
        cs_chg = cs - cs.shift(horizon_months)
        out = self._monthly_to_weekly_causal(cs_chg)
        out.name = "cs_chg"
        return out

    def curve_inversion(self) -> pd.Series:
        """inv_t = max(0, TB3MS - GS10) -- the DEPTH of the yield-curve inversion (3mo
        above 10yr), 0 when the curve is normal. A CAUSAL, P&S-free macro feature.

        WHY ONE-SIDED (clamped at 0), NOT THE RAW SLOPE OR ITS CHANGE: the raw slope LEVEL
        and its change both FLIP SIGN across eras (their bear-meaning reverses: yc_lvl sep
        was -0.58 in 1957-69 but +1.13 in 1983-99), so neither can be a single global
        emission mean. Clamping to inversion DEPTH keeps only the half of the range that
        consistently means "stress": it is non-negative-signed in every era it fires
        (1957-69 +0.35, 1970-82 +0.74) and simply SILENT (0) when the curve is normal and
        not the driver. The equity tell is the inversion itself, not the level.
        COMPLEMENTARITY: this is STRONG exactly where credit is dead -- the rate-driven
        1970-82 bears (+0.74) -- and silent in the default-driven dotcom/GFC bears that
        credit_spread_change covers. corr(cs_chg, inv) ~ 0.05: independent information.
        LIMITATION (accepted): inv fires at/BEFORE a rate-driven top then fades once the
        Fed pivots to cutting and the curve re-steepens mid-bear, so it is an early pulse,
        not a sustained hold -- credit carries the bear BODY, inv carries the rate-driven
        ONSET. Pure inversion depth (no decay memory) is the version validated by era.
        """
        slope = self.macro["GS10"] - self.macro["TB3MS"]  # normal (positive) vs inverted
        inv = (-slope).clip(lower=0.0)  # depth of inversion; 0 when normal
        out = self._monthly_to_weekly_causal(inv)
        out.name = "inv"
        return out

    def drawdown(self, window_weeks: int = 52, reset_pct: float | None = None) -> pd.Series:
        """dd_t = weekly_price_t / (reference peak) - 1.  Causal, price-only (NO labels).

        0 at the reference peak, negative below it. Carries price-LEVEL info the r_t/v_t
        channels lack (they see only price CHANGE), so a model can hold P(bear) through a
        relief RALLY (still deep underwater) rather than reacting to the +returns.

        TWO reference-peak definitions, selected by `reset_pct`:

        * reset_pct is None (default) -> TRAILING-WINDOW peak: max over the last
          `window_weeks`. window=52 (1yr) chosen by a stability sweep (in-bear whipsaw
          crossings halve at 52 vs 39 and plateau; see docs/drawdown_window_sweep.md). A
          fixed calendar window has an unavoidable tension though: too long -> the peak
          CLINGS to the pre-crash high into a recovery (dd stays negative ~9mo past the
          bottom -> recovery false-alarm / lagging bear exit); too short -> the peak
          RATCHETS DOWN following price through a long bear (goes blind mid-bear).

        * reset_pct set (e.g. 0.20) -> EVENT-RESET peak: the reference is the max SINCE the
          last confirmed recovery, and a recovery is CONFIRMED when price has risen
          `reset_pct` off its trailing trough (the textbook +20% bull-market definition).
          This DISSOLVES the window tension instead of trading it: in a bear no reset fires,
          so the peak HOLDS the true pre-crash high for the whole decline (no ratcheting,
          any length); at the confirmed bottom the peak RESETS to the recovery, so dd clears
          to ~0 immediately (no calendar lag). Same "we're climbing out" signal a dd-CHANGE
          channel tried to add, but delivered by fixing dd's REFERENCE -- so it stays ONE
          causal channel, no conditional-independence violation. Verified on the GFC: window
          dd stays ~-0.22 through the 2009 recovery; event-reset dd clears ~9mo sooner.
          20% is a round, literature-standard threshold (a bull = +20% off the low), picked
          for economic meaning, not fit to a metric -- it defines a FEATURE, not the regime;
          the HMM still decides the state by integrating dd with r_t/v_t under learned
          emissions and persistence, and can overrule the reset when r/v disagree.
        """
        if reset_pct is None:
            peak = self.weekly_price.rolling(window_weeks, min_periods=1).max()
            dd = self.weekly_price / peak - 1.0
        else:
            dd = self._event_reset_drawdown(reset_pct)
        dd.name = "dd"
        return dd

    def _event_reset_drawdown(self, reset_pct: float) -> pd.Series:
        """Event-reset drawdown: peak HOLDS while underwater, RESETS on a +reset_pct rally
        off the trailing trough. Causal single left-to-right pass (uses only prices up to t).

        State machine: track the running `peak` (since the last reset) and the `trough`
        beneath it. A new high extends the peak (and resets the trough to it). While below
        the peak, deepen the trough. When price has risen `reset_pct` off that trough while
        still under the peak, a new bull is CONFIRMED -> reset the reference peak to here.
        dd_t = price_t / peak_t - 1.
        """
        p = self.weekly_price.to_numpy()
        n = len(p)
        dd = np.empty(n)
        peak = p[0]
        trough = p[0]
        for t in range(n):
            if p[t] > peak:                      # fresh high: extend peak, reset trough to it
                peak = p[t]
                trough = p[t]
            else:                                # underwater: track the deepest point
                trough = min(trough, p[t])
            # confirmed recovery: +reset_pct off the trough while still below the old peak
            if p[t] < peak and trough > 0 and (p[t] / trough - 1.0) >= reset_pct:
                peak = p[t]                      # reset the reference to the confirmed new bull
                trough = p[t]
            dd[t] = p[t] / peak - 1.0
        return pd.Series(dd, index=self.weekly_price.index)

    def observations(self, include_drawdown: bool = False,
                     drawdown_window_weeks: int = 52,
                     drawdown_reset_pct: float | None = None,
                     include_credit: bool = False,
                     credit_horizon_months: int = 5,
                     include_curve: bool = False) -> pd.DataFrame:
        """Observation frame, aligned and NaN-dropped.

        Default: bivariate (r_t, v_t) -- unchanged, so existing 2-state / plain-3-state
        callers are untouched. Optional extra channels, each appended IN THIS ORDER
        (the model's _OBS_COLS must match):
          * include_drawdown -> dd  (causal price-LEVEL, the relief-rally fix)
          * include_credit   -> cs_chg (BAA-AAA multi-month widening; default-driven bears)
          * include_curve    -> inv (yield-curve inversion depth; rate-driven bears)
        credit/curve are the two COMPLEMENTARY macro channels: credit covers dotcom/GFC-type
        default bears, curve covers the 1970s rate-driven bears where credit is dead (see
        credit_spread_change / curve_inversion). Require macro CSVs loaded (load with
        load_regime_dataset(..., include_macro=True)).

        NaN-drop note: cs_chg's first `credit_horizon_months` are NaN (need lookback) and
        drop a few early weeks, exactly like drawdown's warmup -- causal, expected.
        """
        cols = [self.r_t, self.v_t]
        if include_drawdown:
            cols.append(self.drawdown(drawdown_window_weeks, reset_pct=drawdown_reset_pct))
        if include_credit:
            cols.append(self.credit_spread_change(credit_horizon_months))
        if include_curve:
            cols.append(self.curve_inversion())
        return pd.concat(cols, axis=1).dropna()

    def split(self, train_frac: float = 0.8, include_drawdown: bool = False,
              drawdown_window_weeks: int = 52, drawdown_reset_pct: float | None = None,
              include_credit: bool = False,
              credit_horizon_months: int = 5, include_curve: bool = False,
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological train/test split of the observation frame (section 3: genuine
        out-of-sample tail, no shuffling). Passes through all channel options."""
        obs = self.observations(include_drawdown=include_drawdown,
                                 drawdown_window_weeks=drawdown_window_weeks,
                                 drawdown_reset_pct=drawdown_reset_pct,
                                 include_credit=include_credit,
                                 credit_horizon_months=credit_horizon_months,
                                 include_curve=include_curve)
        n_train = int(len(obs) * train_frac)
        return obs.iloc[:n_train], obs.iloc[n_train:]


def load_regime_dataset(
    start: str = "1990-01-01",
    end: str | None = None,
    include_vix: bool = True,
    include_macro: bool = False,
) -> RegimeDataset:
    """Fetch and assemble the full weekly regime-nowcast dataset.

    include_macro loads the static monthly FRED CSVs (BAA/AAA/GS10/TB3MS) from the
    dataset dir so the credit-spread-change and yield-curve-inversion channels are
    available. Off by default so price-only callers don't need the CSVs present.
    """
    sp500_daily = fetch_daily_closes(SP500_TICKER, start=start, end=end)

    r_t = weekly_log_returns(sp500_daily)
    v_t = weekly_log_realized_vol(sp500_daily)
    weekly_price = sp500_daily.resample(WEEKLY_ANCHOR).last().dropna()

    v_t_vix = None
    if include_vix:
        vix_daily = fetch_daily_closes(VIX_TICKER, start=start, end=end)
        v_t_vix = weekly_log_vix(vix_daily)

    macro = None
    if include_macro:
        macro = {fred_id: _load_macro_series(fred_id) for fred_id in _MACRO_SERIES}

    return RegimeDataset(
        daily_close=sp500_daily,
        weekly_price=weekly_price,
        r_t=r_t,
        v_t=v_t,
        v_t_vix=v_t_vix,
        macro=macro,
    )

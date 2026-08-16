"""Data pipeline for the S&P 500 regime nowcast.

Fetches S&P 500 (and VIX) daily closes, resamples to a weekly observation
series (r_t = weekly log return, v_t = log realized vol from daily returns
within the week), and produces a train/test split.

Nothing in this module looks past the data available at each row's own
week -- the weekly resampling only aggregates *within* that week.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

SP500_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

# Weekly bars are anchored on Friday closes (calendar week).
WEEKLY_ANCHOR = "W-FRI"

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

    daily_close: pd.Series  # S&P 500 daily closes, for the drawdown label
    weekly_price: pd.Series  # last daily close per week, aligned to r_t / v_t index
    r_t: pd.Series  # weekly log return
    v_t: pd.Series  # weekly log realized vol (default v_t channel)
    v_t_vix: pd.Series | None  # weekly log VIX, if fetched (an open question)

    def drawdown(self, reset_pct: float = 0.20) -> pd.Series:
        """dd_t = weekly_price_t / (EVENT-RESET reference peak) - 1.  Causal, price-only.

        0 at the reference peak, negative below it. Carries price-LEVEL info the r_t/v_t
        channels lack (they see only price CHANGE), so a model can hold P(bear) through a
        relief RALLY (still deep underwater) rather than reacting to the +returns.

        EVENT-RESET peak: the reference is the max SINCE the last confirmed recovery, and a
        recovery is CONFIRMED when price has risen `reset_pct` off its trailing trough (the
        textbook +20% bull-market definition). In a bear no reset fires, so the peak HOLDS
        the true pre-crash high for the whole decline (no ratcheting, any length); at the
        confirmed bottom the peak RESETS to the recovery, so dd clears to ~0 immediately (no
        calendar lag). It stays ONE causal channel (no conditional-independence violation).
        Verified on the GFC: a fixed-window dd stayed ~-0.22 through the 2009 recovery;
        event-reset dd clears ~9mo sooner. 20% is a round, literature-standard threshold
        (a bull = +20% off the low), picked for economic meaning, not fit to a metric -- it
        defines a FEATURE, not the regime; the HMM still decides the state by integrating dd
        with r_t/v_t under learned emissions and persistence, and can overrule the reset.

        (A fixed trailing-WINDOW peak was the earlier definition; it was dropped -- the
        window traded onset-lag against recovery-false-alarms with no good setting, which is
        exactly the tension event-reset dissolves.)
        """
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
                     drawdown_reset_pct: float = 0.20) -> pd.DataFrame:
        """Observation frame, aligned and NaN-dropped.

        Base channels: r_t, v_t (weekly log return + log realized vol).

        THIRD CHANNEL -- dd (include_drawdown=True): 
        KEY component of the shipped models. dd is the causal price-LEVEL event-reset drawdown
        that fixes the relief-rally whipsaw and roughly doubles bear recall; the good
        3-state / 4-state configs all turn it ON (see their obs_kwargs). The flag merely
        DEFAULTS off so the bare bivariate 2-state case still works unchanged --
        include_drawdown=False is the legacy/ablation path, not the intended model.
        drawdown_reset_pct is the +% off-trough bull-confirm threshold for the reset peak.
        """
        cols = [self.r_t, self.v_t]
        if include_drawdown:
            cols.append(self.drawdown(reset_pct=drawdown_reset_pct))
        return pd.concat(cols, axis=1).dropna()

    def split(self, train_frac: float = 0.8, include_drawdown: bool = False,
              drawdown_reset_pct: float = 0.20,
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological train/test split of the observation frame (genuine
        out-of-sample tail, no shuffling). Passes through all channel options."""
        obs = self.observations(include_drawdown=include_drawdown,
                                 drawdown_reset_pct=drawdown_reset_pct)
        n_train = int(len(obs) * train_frac)
        return obs.iloc[:n_train], obs.iloc[n_train:]


def load_regime_dataset(
    start: str = "1990-01-01",
    end: str | None = None,
    include_vix: bool = True,
    include_macro: bool = False,
) -> RegimeDataset:
    """Fetch and assemble the full weekly regime-nowcast dataset.

    include_macro is accepted for call-site compatibility but should be False: the macro
    credit/curve channels were removed (see docs/unused_ideas/macro_leading_indicators.md).
    """
    if include_macro:
        raise ValueError(
            "include_macro=True is no longer implemented: the credit-spread / yield-curve "
            "channels were removed."
        )
    sp500_daily = fetch_daily_closes(SP500_TICKER, start=start, end=end)

    r_t = weekly_log_returns(sp500_daily)
    v_t = weekly_log_realized_vol(sp500_daily)
    weekly_price = sp500_daily.resample(WEEKLY_ANCHOR).last().dropna()

    v_t_vix = None
    if include_vix:
        vix_daily = fetch_daily_closes(VIX_TICKER, start=start, end=end)
        v_t_vix = weekly_log_vix(vix_daily)

    return RegimeDataset(
        daily_close=sp500_daily,
        weekly_price=weekly_price,
        r_t=r_t,
        v_t=v_t,
        v_t_vix=v_t_vix,
    )

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Opportunity:
    ticker: str
    action: str  # BUY | SELL SHORT | AVOID
    entry: float
    target: float
    stop: float
    risk_reward: float
    position_size: int  # shares
    best_entry_window: str
    confidence: str  # Low | Medium | High
    score: float
    risk: float
    signal: float  # e.g. directional bias in [-1,1]
    explanation: str
    current_price: float
    timeframes_confirmed: List[str]
    frameworks: List[str]


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _now_et() -> datetime:
    """
    Best-effort US/Eastern without adding deps.
    """
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(tz=ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(tz=timezone.utc)


def entry_window_hint(now: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Return (allowed, hint).
    Avoid first 5 minutes after open and last 15 minutes before close.
    """
    n = now or _now_et()
    t = n.timetz()
    # If we don't have ET reliably, return a generic hint.
    if n.tzinfo is None:
        return (True, "Prefer entries after the open; avoid late-day entries.")

    open_t = time(9, 30, tzinfo=n.tzinfo)
    close_t = time(16, 0, tzinfo=n.tzinfo)
    first_ok = time(9, 35, tzinfo=n.tzinfo)
    last_ok = time(15, 45, tzinfo=n.tzinfo)

    if t < first_ok:
        return (False, "Wait until after 9:35am to avoid the opening volatility.")
    if t > last_ok:
        return (False, "Avoid entries in the last 15 minutes before the close.")

    # Friendly window guidance
    if time(9, 45, tzinfo=n.tzinfo) <= t <= time(10, 15, tzinfo=n.tzinfo):
        return (True, "Enter between 9:45–10:15am if the setup holds.")
    if time(10, 30, tzinfo=n.tzinfo) <= t <= time(11, 30, tzinfo=n.tzinfo):
        return (True, "Good window: 10:30–11:30am on confirmation.")
    if time(12, 0, tzinfo=n.tzinfo) <= t <= time(14, 0, tzinfo=n.tzinfo):
        return (True, "Consider a midday pullback entry (12:00–2:00pm).")
    return (True, "Wait for confirmation; avoid chasing fast moves.")


def _volatility_pct(close: "pd.Series", n: int = 40) -> float:
    import numpy as np

    r = close.pct_change().dropna().tail(n)
    v = float(r.std())
    if np.isnan(v):
        return 0.0
    return v


def _trend_confidence(close: "pd.Series") -> float:
    import numpy as np

    from utils.technical_analysis import ema

    e9 = ema(close, 9)
    e21 = ema(close, 21)
    if len(e9) == 0 or len(e21) == 0:
        return 0.5
    if float(e21.iloc[-1]) == 0:
        return 0.5
    spread = float((e9.iloc[-1] / e21.iloc[-1]) - 1.0)
    c = min(1.0, abs(spread) / 0.02)  # 2% spread ~ high confidence
    if np.isnan(c):
        return 0.5
    return _clip01(c)


def _risk_bucket(vol: float) -> float:
    # Soft risk proxy used for ranking: higher vol => higher risk
    # vol is daily-ish std (e.g., 0.02 = 2%)
    if vol <= 0:
        return 0.3
    if vol < 0.012:
        return 0.35
    if vol < 0.022:
        return 0.55
    if vol < 0.035:
        return 0.7
    return 0.85


def _confidence_label(signal_count: int, trend_conf: float) -> str:
    if signal_count >= 5 and trend_conf >= 0.7:
        return "High"
    if signal_count >= 4:
        return "Medium"
    return "Low"


def macro_context_headlines() -> Dict[str, object]:
    """
    Best-effort macro context using Yahoo Finance news for SPY.
    Adds simple keyword extraction to produce plain-English context.
    """
    keywords = {
        "fed": ["fed", "fomc", "powell", "interest rate", "rates", "yield"],
        "inflation": ["cpi", "inflation", "ppi", "jobs report", "nonfarm", "unemployment"],
        "geopolitics": ["war", "geopolitical", "sanctions", "taiwan", "middle east", "ukraine"],
        "earnings": ["earnings", "guidance", "quarter", "revenue", "profit"],
    }

    headlines: List[str] = []
    try:
        from utils.fundamental_analysis import fetch_recent_headlines_yahoo

        items = fetch_recent_headlines_yahoo("SPY", limit=15)
        headlines = [i.headline for i in items]
    except Exception:
        headlines = []

    hits: List[str] = []
    h_lower = " ".join([h.lower() for h in headlines])
    for k, terms in keywords.items():
        if any(term in h_lower for term in terms):
            if k == "fed":
                hits.append("Fed / rates-related headlines are in focus.")
            elif k == "inflation":
                hits.append("Inflation / macro data headlines are in focus.")
            elif k == "geopolitics":
                hits.append("Geopolitical risk headlines are in focus.")
            elif k == "earnings":
                hits.append("Earnings-related headlines are influencing the tape.")

    note = hits[0] if hits else "No dominant macro theme detected from recent broad-market headlines."
    return {"note": note, "headlines": headlines}


def score_intraday_signals(
    daily: "pd.DataFrame",
    h1: "pd.DataFrame",
    m15: "pd.DataFrame",
    m5: "pd.DataFrame",
) -> Tuple[int, Dict[str, float], str, List[str], str]:
    """
    Compute a set of signals and return:
      - signal_count (how many align)
      - signal_values (for display/debug)
      - explanation fragment (plain English)
    """
    import numpy as np

    from utils.technical_analysis import ema, macd, rsi, scores as pattern_scores

    signals: Dict[str, float] = {}

    # Close series per timeframe
    d_close = daily["Close"].astype(float).dropna()
    h_close = h1["Close"].astype(float).dropna()
    i_close = m15["Close"].astype(float).dropna()
    m_close = m5["Close"].astype(float).dropna()
    if len(d_close) < 60 or len(h_close) < 60 or len(i_close) < 60 or len(m_close) < 60:
        return (0, {}, "Not enough recent data for a reliable multi-timeframe read.", [], "mixed")

    # Trend: EMA9 vs EMA21 on all four timeframes
    d_trend = float((ema(d_close, 9).iloc[-1] / ema(d_close, 21).iloc[-1]) - 1.0)
    h_trend = float((ema(h_close, 9).iloc[-1] / ema(h_close, 21).iloc[-1]) - 1.0)
    i_trend = float((ema(i_close, 9).iloc[-1] / ema(i_close, 21).iloc[-1]) - 1.0)
    m_trend = float((ema(m_close, 9).iloc[-1] / ema(m_close, 21).iloc[-1]) - 1.0)
    signals["trend_daily"] = d_trend
    signals["trend_1h"] = h_trend
    signals["trend_15m"] = i_trend
    signals["trend_5m"] = m_trend

    # Momentum: RSI + MACD hist on 1h and 15m
    h_rsi = float(rsi(h_close, 14).iloc[-1])
    i_rsi = float(rsi(i_close, 14).iloc[-1])
    h_macd_hist = float(macd(h_close).hist.iloc[-1])
    i_macd_hist = float(macd(i_close).hist.iloc[-1])
    signals["rsi_1h"] = h_rsi
    signals["rsi_15m"] = i_rsi
    signals["macd_hist_1h"] = h_macd_hist
    signals["macd_hist_15m"] = i_macd_hist

    # Volume confirmation: last 5m vol vs rolling mean
    vol = m5["Volume"].astype(float).dropna()
    vol_mean = vol.rolling(20, min_periods=5).mean().iloc[-1]
    if vol_mean is None or np.isnan(vol_mean) or vol_mean <= 0:
        vol_mean = 1.0
    vol_ratio = float(vol.iloc[-1] / vol_mean)
    if np.isnan(vol_ratio):
        vol_ratio = 1.0
    signals["volume_ratio_5m"] = vol_ratio

    # Breakout: last 15m close vs 20-bar range
    upper = float(i_close.rolling(20, min_periods=5).max().iloc[-2])
    lower = float(i_close.rolling(20, min_periods=5).min().iloc[-2])
    last = float(i_close.iloc[-1])
    brk_up = 1.0 if last > upper * 1.001 else 0.0
    brk_dn = 1.0 if last < lower * 0.999 else 0.0
    signals["breakout_up"] = brk_up
    signals["breakout_down"] = brk_dn

    # Pattern / SR context from daily
    pats = pattern_scores(daily.tail(260), lookback=min(200, len(daily)))
    signals["sr_score"] = float(pats.get("support_resistance", 0.0))
    signals["flag_score"] = float(pats.get("flags", 0.0))

    # Determine direction preference per timeframe
    bullish_votes = 0
    bearish_votes = 0
    confirmed_timeframes: List[str] = []

    def vote(trend: float, label: str) -> None:
        nonlocal bullish_votes, bearish_votes, confirmed_timeframes
        if trend > 0:
            bullish_votes += 1
            confirmed_timeframes.append(label)
        elif trend < 0:
            bearish_votes += 1
            confirmed_timeframes.append(label)

    vote(d_trend, "1D")
    vote(h_trend, "1H")
    vote(i_trend, "15M")
    vote(m_trend, "5M")

    # Momentum votes
    if h_rsi >= 55:
        bullish_votes += 1
    if h_rsi <= 45:
        bearish_votes += 1
    if i_rsi >= 55:
        bullish_votes += 1
    if i_rsi <= 45:
        bearish_votes += 1
    if h_macd_hist > 0:
        bullish_votes += 1
    if h_macd_hist < 0:
        bearish_votes += 1
    if i_macd_hist > 0:
        bullish_votes += 1
    if i_macd_hist < 0:
        bearish_votes += 1

    # Volume + breakout
    if vol_ratio >= 1.4 and brk_up > 0:
        bullish_votes += 1
    if vol_ratio >= 1.4 and brk_dn > 0:
        bearish_votes += 1

    # SR tilt
    if signals["sr_score"] > 0.35:
        bullish_votes += 1
    if signals["sr_score"] < -0.35:
        bearish_votes += 1

    signal_count = max(bullish_votes, bearish_votes)
    if bullish_votes == bearish_votes:
        direction = "mixed"
    elif bullish_votes > bearish_votes:
        direction = "bullish"
    else:
        direction = "bearish"

    # Plain-English fragment
    parts = []
    parts.append(
        "Multi-timeframe trend and momentum are aligned."
        if direction != "mixed"
        else "Signals are mixed across timeframes."
    )
    if brk_up > 0:
        parts.append("Price is breaking above a recent intraday range with momentum.")
    if brk_dn > 0:
        parts.append("Price is breaking below a recent intraday range with downside momentum.")
    if vol_ratio >= 1.4:
        parts.append("Volume is elevated, supporting the move.")
    if signals["sr_score"] > 0.35:
        parts.append("Daily support/resistance context is supportive.")
    if signals["sr_score"] < -0.35:
        parts.append("Daily resistance is pressuring price.")

    expl = " ".join(parts[:3])
    return (signal_count, signals, expl, confirmed_timeframes, direction)


def make_trade_plan(
    ticker: str,
    *,
    daily: "pd.DataFrame",
    h1: "pd.DataFrame",
    m15: "pd.DataFrame",
    m5: "pd.DataFrame",
    capital: float,
    news_sentiment: float = 0.0,
) -> Opportunity:
    """
    Produce a single intraday opportunity (may be AVOID).
    """
    import numpy as np

    allowed, window_hint = entry_window_hint()

    # Stale signal protection outside regular US session (ET)
    now_et = _now_et()
    if now_et.tzinfo is not None:
        tt = now_et.timetz()
        market_open = time(9, 30, tzinfo=now_et.tzinfo) <= tt <= time(
            16, 0, tzinfo=now_et.tzinfo
        )
    else:
        nt = now_et.time()
        market_open = time(9, 30) <= nt <= time(16, 0)
    if not market_open:
        window_hint = (
            "Market closed. Next open: 9:30am ET. Pre-market signal only — "
            "do not trade until market opens."
        )

    sig_count, sigs, sig_expl, tfs_confirmed, direction = score_intraday_signals(
        daily, h1, m15, m5
    )

    m_close = m5["Close"].astype(float).dropna()
    last = float(m_close.iloc[-1]) if len(m_close) else float("nan")

    # Hard trend filter - daily and 1H must agree
    d_trend_bull = sigs.get("trend_daily", 0) > 0
    d_trend_bear = sigs.get("trend_daily", 0) < 0
    h_trend_bull = sigs.get("trend_1h", 0) > 0
    h_trend_bear = sigs.get("trend_1h", 0) < 0

    if d_trend_bull and h_trend_bull:
        allowed_bias = "long"
    elif d_trend_bear and h_trend_bear:
        allowed_bias = "short"
    else:
        allowed_bias = "none"

    if allowed_bias == "none":
        return Opportunity(
            ticker=ticker,
            action="AVOID",
            entry=float("nan"),
            target=float("nan"),
            stop=float("nan"),
            risk_reward=0.0,
            position_size=0,
            best_entry_window=str(window_hint),
            confidence="Low",
            score=0.0,
            risk=0.0,
            signal=0.0,
            explanation="Avoid: daily and 1H trends conflict — no clear directional bias.",
            current_price=float(last),
            timeframes_confirmed=[],
            frameworks=[],
        )

    # Force bias to match the hard trend filter
    bias = "long" if allowed_bias == "long" else "short"

    i_close = m15["Close"].astype(float).dropna()
    if len(i_close) < 20 or len(m_close) < 20:
        # Not enough intraday history for a precise intraday plan
        return Opportunity(
            ticker=ticker,
            action="AVOID",
            entry=float("nan"),
            target=float("nan"),
            stop=float("nan"),
            risk_reward=0.0,
            position_size=0,
            best_entry_window=str(window_hint),
            confidence="Low",
            score=0.0,
            risk=0.0,
            signal=0.0,
            explanation="Avoid for now: not enough intraday data for a tight intraday setup.",
            current_price=float(last),
            timeframes_confirmed=list(dict.fromkeys(tfs_confirmed)),
            frameworks=[],
        )

    # --- ATR-based stop on 5m ---
    m_high = m5["High"].astype(float).dropna()
    m_low = m5["Low"].astype(float).dropna()
    m_close_all = m5["Close"].astype(float).dropna()
    if len(m_high) < 15 or len(m_low) < 15 or len(m_close_all) < 15:
        # Fallback to simple small stop if ATR cannot be computed
        atr = max(0.005 * last, 0.01)
    else:
        prev_close = m_close_all.shift(1)
        tr1 = (m_high - m_low).abs()
        tr2 = (m_high - prev_close).abs()
        tr3 = (m_low - prev_close).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = float(tr.tail(14).mean())
        if np.isnan(atr) or atr <= 0:
            atr = max(0.005 * last, 0.01)

    high_vol_tickers = {"TSLA", "NVDA", "META", "AMZN", "MSTR", "COIN", "AMD"}
    atr_mult = 1.5 if ticker.upper() in high_vol_tickers else 1.0

    # Entry refinement from latest 5m candle
    last_high = float(m_high.iloc[-1])
    last_low = float(m_low.iloc[-1])
    if bias == "long":
        entry = last_high * 1.0005  # confirm breakout
    else:
        entry = last_low * 0.9995  # confirm breakdown

    # Target distance constraints by instrument/price
    price_ref = entry
    if ticker in ("GC=F", "CL=F"):
        max_target_pct = 0.5  # 0.5%
    else:
        if price_ref < 200:
            max_target_pct = 1.5  # 1.5%
        else:
            max_target_pct = 1.0  # 1.0%
    max_target_dist = price_ref * (max_target_pct / 100.0)

    # Stop distance from ATR, but ensure 2R target stays within max target distance
    stop_dist_raw = atr_mult * atr
    max_stop_dist = max_target_dist / 2.0 if max_target_dist > 0 else stop_dist_raw
    stop_dist = min(stop_dist_raw, max_stop_dist) if max_stop_dist > 0 else stop_dist_raw

    # Fallback if something went wrong
    if stop_dist <= 0 or np.isnan(stop_dist):
        stop_dist = max(0.005 * entry, 0.01)  # 0.5% of entry or 1 cent

    if bias == "long":
        stop = entry - stop_dist
        target = entry + 2.0 * stop_dist
    else:
        stop = entry + stop_dist
        target = entry - 2.0 * stop_dist

    risk_per_share = abs(entry - stop)

    # Final safeguard: ensure target distance still within max_target_dist
    target_dist = abs(target - entry)
    if max_target_dist > 0 and target_dist > max_target_dist:
        # Shrink stop and target proportionally to keep 2:1
        stop_dist = max_target_dist / 2.0
        if bias == "long":
            stop = entry - stop_dist
            target = entry + 2.0 * stop_dist
        else:
            stop = entry + stop_dist
            target = entry - 2.0 * stop_dist
        risk_per_share = abs(entry - stop)

    if risk_per_share <= 0 or np.isnan(risk_per_share):
        # Fallback risk_per_share: 0.5% of entry
        risk_per_share = max(0.005 * entry, 0.01)
        if bias == "long":
            stop = entry - risk_per_share
            target = entry + 2.0 * risk_per_share
        else:
            stop = entry + risk_per_share
            target = entry - 2.0 * risk_per_share

    rr = (abs(target - entry) / risk_per_share) if risk_per_share > 0 else 0.0

    # Position sizing: risk 1% of capital with robust fallback
    risk_budget = max(0.0, float(capital) * 0.01)
    shares = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    if (shares <= 0 or np.isnan(shares)) and capital > 500:
        shares = 1

    vol = _volatility_pct(daily["Close"].astype(float))
    trend_conf = _trend_confidence(i_close)
    risk = _risk_bucket(vol)

    # Strategy framework checks
    frameworks: List[str] = []

    from utils.technical_analysis import sma

    # Mark Minervini style: near 52-week high, above 50/200 MA, tight base, volume surge
    d_close = daily["Close"].astype(float).dropna()
    d_vol = daily["Volume"].astype(float).dropna()
    if len(d_close) >= 200 and len(d_vol) >= 40:
        high_52w = float(d_close.tail(252).max())
        last_close = float(d_close.iloc[-1])
        sma50 = float(sma(d_close, 50).iloc[-1])
        sma200 = float(sma(d_close, 200).iloc[-1])
        base_range = float(d_close.tail(20).max() - d_close.tail(20).min())
        base_mid = float((d_close.tail(20).max() + d_close.tail(20).min()) / 2.0)
        base_tight = base_mid > 0 and base_range / base_mid <= 0.08
        near_high = high_52w > 0 and last_close >= high_52w * 0.9
        vol_surge = float(d_vol.iloc[-1]) > float(d_vol.tail(20).mean()) * 1.5
        above_mas = last_close > sma50 and last_close > sma200
        if near_high and base_tight and vol_surge and above_mas:
            frameworks.append("Mark Minervini")

    # Paul Tudor Jones: only trade with dominant daily trend
    dom_trend_long = direction == "bullish"
    dom_trend_short = direction == "bearish"
    if (bias == "long" and dom_trend_long) or (bias == "short" and dom_trend_short):
        frameworks.append("Paul Tudor Jones")

    # VWAP mean reversion heuristic on 5m
    m_close = m5["Close"].astype(float).dropna()
    m_vol = m5["Volume"].astype(float).dropna()
    if len(m_close) >= 20 and len(m_vol) >= 20:
        cum_vol = m_vol.cumsum()
        cum_pv = (m_close * m_vol).cumsum()
        vwap = (cum_pv / cum_vol).iloc[-1]
        last_price = float(m_close.iloc[-1])
        if vwap > 0:
            dist = abs(last_price - vwap) / vwap
            if dist >= 0.015:
                # if bias is to revert towards VWAP
                if bias == "long" and last_price < vwap:
                    frameworks.append("VWAP mean reversion")
                if bias == "short" and last_price > vwap:
                    frameworks.append("VWAP mean reversion")

    # Opening range breakout on 15m
    if len(m15) >= 4:
        orange = m15.iloc[:2]
        or_high = float(orange["High"].max())
        or_low = float(orange["Low"].min())
        last_close_15 = float(m15["Close"].iloc[-1])
        vol15 = m15["Volume"].astype(float)
        v_spike = float(vol15.iloc[-1]) > float(vol15.tail(20).mean()) * 1.5
        broke_up = last_close_15 > or_high * 1.001
        broke_dn = last_close_15 < or_low * 0.999
        if v_spike and ((bias == "long" and broke_up) or (bias == "short" and broke_dn)):
            frameworks.append("Opening range breakout")

    # ICT-style liquidity sweep / FVG heuristic on 5m/15m
    def _has_simple_sweep(df_ctx: "pd.DataFrame") -> bool:
        c = df_ctx["Close"].astype(float).dropna()
        h = df_ctx["High"].astype(float).dropna()
        l = df_ctx["Low"].astype(float).dropna()
        if len(c) < 5:
            return False
        # last bar wicks beyond recent high/low and closes back inside
        recent_h = float(h.iloc[-5:-1].max())
        recent_l = float(l.iloc[-5:-1].min())
        last_h = float(h.iloc[-1])
        last_l = float(l.iloc[-1])
        last_c = float(c.iloc[-1])
        swept_high = last_h > recent_h and last_c < recent_h
        swept_low = last_l < recent_l and last_c > recent_l
        return swept_high or swept_low

    has_ict = _has_simple_sweep(m5) or _has_simple_sweep(m15)
    if has_ict:
        frameworks.append("ICT liquidity / FVG")

    # Paul Tudor Jones alone is not enough - require at least one technically specific framework
    specific_frameworks = {
        "Mark Minervini",
        "VWAP mean reversion",
        "Opening range breakout",
        "ICT liquidity / FVG",
    }
    if not any(f in specific_frameworks for f in frameworks):
        frameworks = []

    # Confluence rule: higher timeframes must agree, and at least one lower timeframe confirms
    timeframe_signals = [
        sigs.get("trend_daily", 0),
        sigs.get("trend_1h", 0),
        sigs.get("trend_15m", 0),
        sigs.get("trend_5m", 0),
    ]
    daily_h1_agree_bull = timeframe_signals[0] > 0 and timeframe_signals[1] > 0
    daily_h1_agree_bear = timeframe_signals[0] < 0 and timeframe_signals[1] < 0
    lower_tf_confirms = (
        (timeframe_signals[2] > 0 or timeframe_signals[3] > 0)
        if daily_h1_agree_bull
        else (timeframe_signals[2] < 0 or timeframe_signals[3] < 0)
    )
    confluence_ok = (daily_h1_agree_bull or daily_h1_agree_bear) and lower_tf_confirms

    quality_frameworks = [
        f
        for f in frameworks
        if f in ("Mark Minervini", "VWAP mean reversion", "Opening range breakout")
    ]
    # Allow ICT to qualify for medium-grade intraday setups if higher-timeframe confluence is strong.
    has_quality_framework = len(quality_frameworks) >= 1 or (
        "ICT liquidity / FVG" in frameworks
    )

    action = "AVOID"
    core_setup_ok = (
        confluence_ok
        and sig_count >= 4
        and has_quality_framework
        and len(frameworks) >= 1
        and rr >= 2.0
        and shares > 0
    )
    if core_setup_ok:
        # During non-entry windows, still return directional pre-session setup (not forced AVOID).
        action = "BUY" if bias == "long" else "SELL SHORT"

    # Confidence tiers: high requires stronger confluence and >=2 frameworks
    if action != "AVOID" and sig_count >= 5 and len(frameworks) >= 2 and allowed:
        conf = "High"
    elif action != "AVOID":
        conf = "Medium"
    else:
        conf = _confidence_label(sig_count, trend_conf)

    # A simple combined score for ranking
    dir_signal = 1.0 if action == "BUY" else (-1.0 if action == "SELL SHORT" else 0.0)
    signal_strength = min(1.0, sig_count / 6.0)
    score = (signal_strength * (0.6 + 0.4 * trend_conf)) * (1.0 if action != "AVOID" else 0.4)
    score += 0.15 * float(news_sentiment) * dir_signal
    score -= 0.35 * risk

    # Explanation (2–3 sentences max)
    rr_txt = f"Risk/reward is ~{rr:.1f}:1 with a 1% risk position size of {shares} shares."
    if action == "AVOID":
        why = "Avoid for now: the setup doesn’t meet the minimum alignment and risk/reward rules."
        if not allowed:
            why = "Avoid for now: it’s outside the preferred entry times around the open/close."
        if rr < 2.0:
            why = "Avoid for now: the expected risk/reward is below 2:1."
        explanation = f"{why} {window_hint}"
    else:
        direction_phrase = "Upside" if action == "BUY" else "Downside"
        if allowed:
            explanation = f"{sig_expl} {direction_phrase} bias is supported today. {rr_txt}"
        else:
            explanation = (
                f"Pre-session setup: {sig_expl} {direction_phrase} bias is present, "
                f"but entry timing is not currently in the preferred window. {window_hint}"
            )

    return Opportunity(
        ticker=ticker,
        action=action,
        entry=float(entry),
        target=float(target),
        stop=float(stop),
        risk_reward=float(rr),
        position_size=int(shares),
        best_entry_window=str(window_hint),
        confidence=str(conf),
        score=float(score),
        risk=float(risk),
        signal=float(dir_signal),
        explanation=explanation.strip(),
        current_price=float(last),
        timeframes_confirmed=list(dict.fromkeys(tfs_confirmed)),
        frameworks=frameworks,
    )


def as_dicts(opps: List[Opportunity]) -> List[Dict[str, object]]:
    return [asdict(o) for o in opps]


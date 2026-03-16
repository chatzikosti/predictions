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
    signal: float  # e.g. p_up or directional signal in [-1,1]
    explanation: str
    current_price: float


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
    m15: "pd.DataFrame",
    m5: "pd.DataFrame",
) -> Tuple[int, Dict[str, float], str]:
    """
    Compute a set of signals and return:
      - signal_count (how many align)
      - signal_values (for display/debug)
      - explanation fragment (plain English)
    """
    import numpy as np

    from utils.technical_analysis import ema, macd, rsi, scores as pattern_scores

    signals: Dict[str, float] = {}

    # Trend: EMA9 vs EMA21 on 15m and daily
    d_close = daily["Close"].astype(float).dropna()
    i_close = m15["Close"].astype(float).dropna()
    m_close = m5["Close"].astype(float).dropna()
    if len(d_close) < 30 or len(i_close) < 50 or len(m_close) < 50:
        return (0, {}, "Not enough recent data for a reliable intraday read.")

    d_trend = float((ema(d_close, 9).iloc[-1] / ema(d_close, 21).iloc[-1]) - 1.0)
    i_trend = float((ema(i_close, 9).iloc[-1] / ema(i_close, 21).iloc[-1]) - 1.0)
    signals["trend_daily"] = d_trend
    signals["trend_15m"] = i_trend

    # Momentum: RSI + MACD hist on 15m
    i_rsi = float(rsi(i_close, 14).iloc[-1])
    i_macd_hist = float(macd(i_close).hist.iloc[-1])
    signals["rsi_15m"] = i_rsi
    signals["macd_hist_15m"] = i_macd_hist

    # Volume confirmation: last 5m vol vs rolling mean
    vol = m5["Volume"].astype(float).dropna()
    vol_ratio = float(vol.iloc[-1] / (vol.rolling(20, min_periods=20).mean().iloc[-1] or np.nan))
    signals["volume_ratio_5m"] = vol_ratio

    # Breakout: last 15m close vs 20-bar range
    upper = float(i_close.rolling(20, min_periods=20).max().iloc[-2])
    lower = float(i_close.rolling(20, min_periods=20).min().iloc[-2])
    last = float(i_close.iloc[-1])
    brk_up = 1.0 if last > upper * 1.001 else 0.0
    brk_dn = 1.0 if last < lower * 0.999 else 0.0
    signals["breakout_up"] = brk_up
    signals["breakout_down"] = brk_dn

    # Pattern / SR context from daily
    pats = pattern_scores(daily.tail(260), lookback=min(200, len(daily)))
    signals["sr_score"] = float(pats.get("support_resistance", 0.0))
    signals["flag_score"] = float(pats.get("flags", 0.0))

    # Determine direction preference
    bullish_votes = 0
    bearish_votes = 0

    # Trend votes
    if d_trend > 0:
        bullish_votes += 1
    if d_trend < 0:
        bearish_votes += 1
    if i_trend > 0:
        bullish_votes += 1
    if i_trend < 0:
        bearish_votes += 1

    # Momentum votes
    if i_rsi >= 55:
        bullish_votes += 1
    if i_rsi <= 45:
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
    parts.append("Short-term trend and momentum are aligned." if direction != "mixed" else "Signals are mixed.")
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
    return (signal_count, signals, expl)


def make_trade_plan(
    ticker: str,
    *,
    daily: "pd.DataFrame",
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
    sig_count, sigs, sig_expl = score_intraday_signals(daily, m15, m5)

    # Directional bias from 15m trend
    from utils.technical_analysis import ema

    i_close = m15["Close"].astype(float).dropna()
    m_close = m5["Close"].astype(float).dropna()
    last = float(m_close.iloc[-1])
    e9 = float(ema(i_close, 9).iloc[-1])
    e21 = float(ema(i_close, 21).iloc[-1])
    bias = "long" if e9 >= e21 else "short"

    # Stops from recent 15m range
    recent_low = float(i_close.rolling(20, min_periods=20).min().iloc[-1])
    recent_high = float(i_close.rolling(20, min_periods=20).max().iloc[-1])

    entry = last
    if bias == "long":
        stop = min(recent_low, entry * 0.995)
        risk_per_share = max(0.0, entry - stop)
        target = entry + 2.0 * risk_per_share
        rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else 0.0
    else:
        stop = max(recent_high, entry * 1.005)
        risk_per_share = max(0.0, stop - entry)
        target = entry - 2.0 * risk_per_share
        rr = (entry - target) / (stop - entry) if (stop - entry) > 0 else 0.0

    # Position sizing: risk 1% of capital
    risk_budget = max(0.0, float(capital) * 0.01)
    shares = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

    vol = _volatility_pct(daily["Close"].astype(float))
    trend_conf = _trend_confidence(i_close)
    risk = _risk_bucket(vol)

    # Only trade if >=3 signals align AND allowed time window AND RR>=2
    action = "AVOID"
    if sig_count >= 3 and allowed and rr >= 2.0 and shares > 0:
        action = "BUY" if bias == "long" else "SELL SHORT"

    # Confidence
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
        explanation = f"{sig_expl} {direction_phrase} bias is supported today. {rr_txt}"

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
    )


def as_dicts(opps: List[Opportunity]) -> List[Dict[str, object]]:
    return [asdict(o) for o in opps]


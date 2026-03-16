from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _compute_risk_metrics(ohlcv: "pd.DataFrame") -> "RiskMetrics":
    import numpy as np
    import pandas as pd

    from utils.technical_analysis import ema
    from utils.recommendation import RiskMetrics

    close = ohlcv["Close"].astype(float).dropna()
    if len(close) < 60:
        # fallback: not much history
        vol = float(close.pct_change().std()) if len(close) > 5 else 0.0
        return RiskMetrics(volatility=vol, trend_confidence=0.5)

    # Volatility: daily return std over last 20 days
    rets = close.pct_change()
    vol = float(rets.tail(20).std())

    # Trend confidence: normalized EMA(12)/EMA(26) spread magnitude (capped)
    e12 = ema(close, 12)
    e26 = ema(close, 26)
    trend = float((e12.iloc[-1] / e26.iloc[-1]) - 1.0) if e26.iloc[-1] else 0.0
    # map ~0-4% spread into 0-1 confidence
    tc = _clip01(min(1.0, abs(trend) / 0.04))

    if np.isnan(vol):
        vol = 0.0
    if np.isnan(tc):
        tc = 0.5
    return RiskMetrics(volatility=vol, trend_confidence=tc)


def get_predictions(tickers: str = "AAPL,MSFT,NVDA,TSLA,AMZN", period: str = "6mo"):
    """
    End-to-end pipeline used by both CLI and UI.

    Returns a dict with:
      - tickers, period
      - technical_summary, sentiment_by_ticker, predictions, risk_by_ticker, recommendations
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        print("[error] Missing dependency: pandas")
        print("Install: pip install pandas numpy yfinance")
        raise

    from utils.data_fetch import fetch_ohlcv, split_by_ticker
    from utils.recommendation import recommend_top3
    from utils.technical_analysis import (
        bollinger_bands,
        macd,
        rsi,
        scores as pattern_scores,
        sma,
        ema,
    )

    tickers_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        raise ValueError("No tickers provided.")

    # 1) Fetch stock data
    df = fetch_ohlcv(tickers_list, period=period)
    ohlcv_by_ticker = (
        split_by_ticker(df)
        if isinstance(df.columns, pd.MultiIndex)
        else {tickers_list[0]: df}
    )

    # 2) Technical analysis
    technical_summary: Dict[str, Dict[str, float]] = {}
    for t, d in ohlcv_by_ticker.items():
        close = d["Close"].astype(float)
        sma10 = float(sma(close, 10).iloc[-1])
        sma50 = float(sma(close, 50).iloc[-1]) if len(close) >= 50 else float("nan")
        rsi14 = float(rsi(close, 14).iloc[-1])
        macd_res = macd(close)
        macd_hist = float(macd_res.hist.iloc[-1])
        bb = bollinger_bands(close)
        bb_pctb = float(((close.iloc[-1] - bb.lower.iloc[-1]) / (bb.upper.iloc[-1] - bb.lower.iloc[-1])) if (bb.upper.iloc[-1] - bb.lower.iloc[-1]) else float("nan"))

        pats = pattern_scores(d, lookback=min(200, len(d)))
        technical_summary[t] = {
            "close": float(close.iloc[-1]),
            "sma10": sma10,
            "sma50": sma50,
            "rsi14": rsi14,
            "macd_hist": macd_hist,
            "bb_percent_b": bb_pctb,
            **{f"pat_{k}": float(v) for k, v in pats.items()},
        }

    # 3) Fundamental analysis (news sentiment)
    sentiment_by_ticker: Dict[str, float] = {t: 0.0 for t in ohlcv_by_ticker.keys()}
    try:
        from utils.fundamental_analysis import ticker_news_sentiment

        sentiment_by_ticker = ticker_news_sentiment(list(ohlcv_by_ticker.keys()), source="yahoo")
    except Exception as e:
        print(f"[warn] Fundamental sentiment unavailable, using 0.0. Reason: {e}")

    # 4) Predictions
    predictions: Dict[str, Dict[str, float]] = {}
    try:
        from models.predictor import NextDayMovementPredictor

        for t, d in ohlcv_by_ticker.items():
            sent = float(sentiment_by_ticker.get(t, 0.0))
            model = NextDayMovementPredictor().fit(d, sentiment_score=sent)
            p_up = float(model.predict_proba_up(d, sentiment_score=sent))
            predictions[t] = {"p_up": p_up}
    except Exception as e:
        print(f"[warn] Predictions unavailable (missing deps?), using neutral 0.50. Reason: {e}")
        for t in ohlcv_by_ticker.keys():
            predictions[t] = {"p_up": 0.50}

    # Risk metrics for recommendation step
    risk_by_ticker = {t: _compute_risk_metrics(d) for t, d in ohlcv_by_ticker.items()}

    # 5) Suggest top 3 positions
    recs = recommend_top3(predictions, risk_by_ticker)

    return {
        "tickers": list(ohlcv_by_ticker.keys()),
        "period": period,
        "technical_summary": technical_summary,
        "sentiment_by_ticker": sentiment_by_ticker,
        "predictions": predictions,
        "risk_by_ticker": {t: asdict(r) for t, r in risk_by_ticker.items()},
        "recommendations": [asdict(r) for r in recs],
    }


def main() -> None:
    print("=== Stock Predictions ===")
    results = get_predictions()

    tickers = results["tickers"]
    period = results["period"]
    technical_summary = results["technical_summary"]
    sentiment_by_ticker = results["sentiment_by_ticker"]
    predictions = results["predictions"]
    risk_by_ticker = results["risk_by_ticker"]
    recs = results["recommendations"]

    print(f"Tickers: {', '.join(tickers)} | Period: {period}")

    print("\n=== Per-ticker summary ===")
    for t in sorted(tickers):
        tech = technical_summary[t]
        sent = float(sentiment_by_ticker.get(t, 0.0))
        p_up = float(predictions[t]["p_up"])
        risk = risk_by_ticker[t]
        print(f"\n[{t}]")
        print(f"  Close: {tech['close']:.2f}")
        print(f"  Prediction: p(up tomorrow)={p_up:.2f}")
        print(f"  News sentiment: {sent:+.2f}")
        print(
            f"  Risk: volatility≈{risk['volatility']*100:.2f}%/day | "
            f"trend_conf≈{risk['trend_confidence']:.2f}"
        )
        print(
            f"  Indicators: SMA10={tech['sma10']:.2f} | SMA50={tech['sma50']:.2f} | "
            f"RSI14={tech['rsi14']:.1f} | MACD_hist={tech['macd_hist']:+.4f} | "
            f"BB_%B={tech['bb_percent_b']:.2f}"
        )
        print(
            "  Patterns:"
            f" H&S={tech['pat_head_shoulders']:+.2f},"
            f" Double={tech['pat_double_top_bottom']:+.2f},"
            f" Triangles={tech['pat_triangles']:+.2f},"
            f" Flags={tech['pat_flags']:+.2f},"
            f" S/R={tech['pat_support_resistance']:+.2f}"
        )

    print("\n=== Top 3 positions (risk-adjusted) ===")
    if not recs:
        print("No recommendations (missing risk metrics or predictions).")
    else:
        for i, r in enumerate(recs, 1):
            print(f"\n#{i} {r['ticker']} -> {r['action'].upper()}")
            print(f"  score={r['score']:+.3f} | risk={r['risk']:.2f}")
            print(f"  why: {r['explanation']}")


if __name__ == "__main__":
    main()

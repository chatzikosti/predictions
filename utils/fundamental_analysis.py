from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class NewsItem:
    ticker: str
    headline: str
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class HeadlineSentiment:
    headline: str
    score: float  # [-1, 1]
    label: str
    confidence: float  # [0, 1]


def _require_yfinance() -> None:
    try:
        import yfinance as yf  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency. Install with: pip install yfinance"
        ) from e


def _require_requests() -> None:
    try:
        import requests  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency. Install with: pip install requests"
        ) from e


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency. Install with: pip install transformers torch"
        ) from e


def _as_ticker_list(tickers: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(tickers, str):
        parts = [t.strip() for t in tickers.replace(";", ",").split(",")]
        out = [p for p in parts if p]
    else:
        out = [str(t).strip() for t in tickers if str(t).strip()]

    seen = set()
    uniq: List[str] = []
    for t in out:
        key = t.upper()
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    if not uniq:
        raise ValueError("tickers must contain at least one symbol")
    return uniq


def fetch_recent_headlines_yahoo(
    ticker: str,
    *,
    limit: int = 20,
) -> List[NewsItem]:
    """
    Fetch recent headlines from Yahoo Finance via yfinance.

    Note: yfinance's news coverage varies by ticker/region.
    """
    _require_yfinance()
    import yfinance as yf

    t = yf.Ticker(ticker)
    items = []
    raw = getattr(t, "news", None) or []
    for n in raw[: max(limit, 0)]:
        title = str(n.get("title") or "").strip()
        if not title:
            continue
        link = n.get("link")
        src = None
        if isinstance(n.get("publisher"), str):
            src = n.get("publisher")
        elif isinstance(n.get("source"), str):
            src = n.get("source")

        # yfinance sometimes provides epoch seconds in "providerPublishTime"
        published_at = None
        ts = n.get("providerPublishTime")
        if isinstance(ts, (int, float)) and ts > 0:
            published_at = datetime.fromtimestamp(ts, tz=timezone.utc)

        items.append(
            NewsItem(
                ticker=ticker,
                headline=title,
                url=str(link) if link else None,
                published_at=published_at,
                source=src,
            )
        )
    return items


def fetch_recent_headlines_newsapi(
    ticker_or_query: str,
    *,
    api_key: Optional[str] = None,
    limit: int = 20,
    language: str = "en",
    sort_by: str = "publishedAt",
) -> List[NewsItem]:
    """
    Fetch recent headlines from NewsAPI.

    You can pass an API key explicitly, or set env var NEWSAPI_KEY.
    """
    _require_requests()
    import requests

    key = api_key or os.getenv("NEWSAPI_KEY")
    if not key:
        raise ValueError("NewsAPI key missing. Provide api_key=... or set NEWSAPI_KEY.")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker_or_query,
        "language": language,
        "sortBy": sort_by,
        "pageSize": min(max(int(limit), 1), 100),
        "apiKey": key,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    items: List[NewsItem] = []
    for a in (data.get("articles") or [])[: max(limit, 0)]:
        title = str(a.get("title") or "").strip()
        if not title:
            continue
        src = None
        source = a.get("source") or {}
        if isinstance(source, dict) and isinstance(source.get("name"), str):
            src = source.get("name")

        published_at = None
        ts = a.get("publishedAt")
        if isinstance(ts, str) and ts:
            try:
                # ISO-8601, often ends with 'Z'
                s = ts.replace("Z", "+00:00")
                published_at = datetime.fromisoformat(s)
            except ValueError:
                published_at = None

        items.append(
            NewsItem(
                ticker=ticker_or_query,
                headline=title,
                url=str(a.get("url")) if a.get("url") else None,
                published_at=published_at,
                source=src,
            )
        )
    return items


_PIPELINE = None
_PIPELINE_MODEL = None


def _get_sentiment_pipeline(model: Optional[str] = None):
    """
    Lazily construct a HuggingFace sentiment pipeline.

    Defaults to FinBERT (finance-focused) if model not provided.
    """
    global _PIPELINE, _PIPELINE_MODEL
    _require_transformers()
    from transformers import pipeline

    chosen = model or "ProsusAI/finbert"
    if _PIPELINE is None or _PIPELINE_MODEL != chosen:
        _PIPELINE = pipeline("sentiment-analysis", model=chosen)
        _PIPELINE_MODEL = chosen
    return _PIPELINE


def _label_to_signed_score(label: str, confidence: float) -> float:
    """
    Map pipeline label to signed score in [-1, 1].
    Supports common label sets:
    - POSITIVE / NEGATIVE / NEUTRAL (FinBERT-style)
    - POSITIVE / NEGATIVE (SST-2-style)
    """
    lab = (label or "").strip().upper()
    if lab in ("POSITIVE", "LABEL_1"):
        return float(confidence)
    if lab in ("NEGATIVE", "LABEL_0"):
        return -float(confidence)
    if lab in ("NEUTRAL",):
        return 0.0
    # Unknown label: treat as neutral
    return 0.0


def score_headlines_sentiment(
    headlines: Sequence[str],
    *,
    model: Optional[str] = None,
    batch_size: int = 16,
    truncate: bool = True,
) -> List[HeadlineSentiment]:
    """
    Score each headline with a pretrained HF sentiment model.
    Returns list of HeadlineSentiment with signed scores in [-1, 1].
    """
    if not headlines:
        return []

    pipe = _get_sentiment_pipeline(model=model)

    results: List[HeadlineSentiment] = []
    for i in range(0, len(headlines), max(int(batch_size), 1)):
        batch = [h for h in headlines[i : i + batch_size] if str(h).strip()]
        if not batch:
            continue
        outs = pipe(batch, truncation=truncate)
        for h, o in zip(batch, outs):
            label = str(o.get("label") or "")
            conf = float(o.get("score") or 0.0)
            signed = _label_to_signed_score(label, conf)
            results.append(
                HeadlineSentiment(
                    headline=str(h),
                    score=signed,
                    label=label,
                    confidence=conf,
                )
            )
    return results


def aggregate_sentiment(
    scored: Sequence[HeadlineSentiment],
    *,
    method: str = "mean",
) -> float:
    """
    Combine headline sentiments into a single score in [-1, 1].

    method:
    - "mean": average signed scores
    - "confidence_weighted": average of signed scores weighted by confidence
    """
    if not scored:
        return 0.0

    if method == "mean":
        return float(sum(s.score for s in scored) / len(scored))

    if method == "confidence_weighted":
        wsum = sum(max(0.0, min(1.0, s.confidence)) for s in scored)
        if wsum == 0:
            return 0.0
        return float(
            sum(s.score * max(0.0, min(1.0, s.confidence)) for s in scored) / wsum
        )

    raise ValueError("method must be 'mean' or 'confidence_weighted'")


def ticker_news_sentiment(
    tickers: Union[str, Iterable[str]],
    *,
    source: str = "yahoo",
    limit_per_ticker: int = 20,
    model: Optional[str] = None,
    aggregate: str = "confidence_weighted",
    newsapi_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    End-to-end helper:
    - fetch recent headlines per ticker
    - score sentiment using a pretrained model
    - aggregate to a single score per ticker
    """
    tlist = _as_ticker_list(tickers)
    out: Dict[str, float] = {}

    for t in tlist:
        if source.lower() == "yahoo":
            items = fetch_recent_headlines_yahoo(t, limit=limit_per_ticker)
        elif source.lower() == "newsapi":
            items = fetch_recent_headlines_newsapi(
                t, api_key=newsapi_key, limit=limit_per_ticker
            )
        else:
            raise ValueError("source must be 'yahoo' or 'newsapi'")

        headlines = [n.headline for n in items]
        scored = score_headlines_sentiment(headlines, model=model)
        out[t] = aggregate_sentiment(scored, method=aggregate)

    return out


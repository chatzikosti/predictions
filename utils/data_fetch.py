from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Literal, Mapping, Optional, Union, overload


DateLike = Union[str, date, datetime]
Interval = Literal[
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
]


@dataclass(frozen=True)
class FetchConfig:
    """
    Configuration for OHLCV downloads.

    Notes:
    - yfinance returns OHLCV columns (plus Adj Close if auto_adjust=False).
    - For multiple tickers, yfinance returns a DataFrame with multi-index columns by default.
    """

    interval: Interval = "1d"
    auto_adjust: bool = False
    prepost: bool = False
    actions: bool = False
    threads: bool = True
    rounding: bool = False
    progress: bool = False
    timeout_s: Optional[float] = 30.0


def _as_ticker_list(tickers: Union[str, Iterable[str]]) -> list[str]:
    if isinstance(tickers, str):
        parts = [t.strip() for t in tickers.replace(";", ",").split(",")]
        out = [p for p in parts if p]
    else:
        out = [str(t).strip() for t in tickers if str(t).strip()]
    # de-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        key = t.upper()
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    if not uniq:
        raise ValueError("tickers must contain at least one symbol")
    return uniq


def _require_deps():
    try:
        import pandas as pd  # noqa: F401
        import yfinance as yf  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency. Install with: pip install yfinance pandas"
        ) from e


@overload
def fetch_ohlcv(
    tickers: str,
    *,
    start: Optional[DateLike] = ...,
    end: Optional[DateLike] = ...,
    period: Optional[str] = ...,
    config: Optional[FetchConfig] = ...,
    keep_only_ohlcv: bool = ...,
) -> "pd.DataFrame": ...


@overload
def fetch_ohlcv(
    tickers: Iterable[str],
    *,
    start: Optional[DateLike] = ...,
    end: Optional[DateLike] = ...,
    period: Optional[str] = ...,
    config: Optional[FetchConfig] = ...,
    keep_only_ohlcv: bool = ...,
) -> "pd.DataFrame": ...


def fetch_ohlcv(
    tickers: Union[str, Iterable[str]],
    *,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    period: Optional[str] = None,
    config: Optional[FetchConfig] = None,
    keep_only_ohlcv: bool = True,
):
    """
    Fetch historical OHLCV for one or many tickers via yfinance.

    Returns:
    - Single ticker: DataFrame indexed by datetime, columns: Open/High/Low/Close/Volume
    - Multiple tickers: DataFrame indexed by datetime, **multi-index columns**:
        level 0 = field (Open/High/Low/Close/Volume)
        level 1 = ticker

    Args:
    - tickers: "AAPL" or ["AAPL","MSFT"] (also accepts "AAPL,MSFT")
    - start/end: date-like bounds (optional)
    - period: e.g. "1y", "5d" (optional; yfinance uses either period OR start/end)
    - config: FetchConfig for interval/adjustments/etc.
    - keep_only_ohlcv: drop non-OHLCV columns (e.g. Adj Close) if present
    """

    _require_deps()
    import pandas as pd
    import yfinance as yf

    cfg = config or FetchConfig()
    ticker_list = _as_ticker_list(tickers)

    if period is None and start is None and end is None:
        # yfinance requires either period or a date range; default to a sane window.
        period = "1y"

    # group_by="column" yields columns like:
    # - single ticker: Open, High, Low, Close, Adj Close, Volume
    # - multi tickers: MultiIndex with first level field, second level ticker
    df = yf.download(
        tickers=" ".join(ticker_list),
        start=start,
        end=end,
        period=period,
        interval=cfg.interval,
        group_by="column",
        auto_adjust=cfg.auto_adjust,
        prepost=cfg.prepost,
        actions=cfg.actions,
        threads=cfg.threads,
        rounding=cfg.rounding,
        progress=cfg.progress,
        timeout=cfg.timeout_s,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for tickers={ticker_list!r}")

    # Normalize index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Keep only OHLCV fields by default
    if keep_only_ohlcv:
        wanted = {"Open", "High", "Low", "Close", "Volume"}
        if isinstance(df.columns, pd.MultiIndex):
            df = df[[c for c in df.columns if c[0] in wanted]]
        else:
            df = df[[c for c in df.columns if c in wanted]]

    # Ensure a stable column order where possible
    if isinstance(df.columns, pd.MultiIndex):
        order = ["Open", "High", "Low", "Close", "Volume"]
        fields = [f for f in order if f in set(df.columns.get_level_values(0))]
        tickers_present = list(dict.fromkeys(df.columns.get_level_values(1)))
        desired = [(f, t) for f in fields for t in tickers_present if (f, t) in df.columns]
        df = df[desired]
    else:
        order = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[order]

    return df


def split_by_ticker(df: "pd.DataFrame") -> Mapping[str, "pd.DataFrame"]:
    """
    Convenience helper: if df has MultiIndex columns (field, ticker),
    return a dict: {ticker -> DataFrame[Open/High/Low/Close/Volume]}.
    """

    _require_deps()
    import pandas as pd

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("split_by_ticker expects a DataFrame with MultiIndex columns")

    out: dict[str, pd.DataFrame] = {}
    for t in dict.fromkeys(df.columns.get_level_values(1)):
        sub = df.xs(t, axis=1, level=1, drop_level=True).copy()
        out[str(t)] = sub
    return out


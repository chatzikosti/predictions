from __future__ import annotations

from typing import List, Tuple


def get_sp500_tickers() -> Tuple[List[str], str]:
    """
    Return (tickers, note).

    Attempts to fetch the live S&P 500 constituents from Wikipedia via pandas.read_html.
    If unavailable (no internet / missing pandas), returns a small fallback list.
    """
    try:
        import pandas as pd
    except Exception:
        return (
            ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH"],
            "Fallback universe (pandas not available).",
        )

    try:
        # Wikipedia table: "List of S&P 500 companies"
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        # First table contains constituents; column is "Symbol"
        df = tables[0]
        syms = [str(s).strip().replace(".", "-") for s in df["Symbol"].tolist()]
        syms = [s for s in syms if s]
        return (syms, "Universe: live S&P 500 constituents (Wikipedia).")
    except Exception:
        return (
            ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH"],
            "Fallback universe (Wikipedia fetch unavailable).",
        )


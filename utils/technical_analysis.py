from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Public convention used throughout:
# - scores are in [-1, 1]
# - positive => bullish bias, negative => bearish bias, 0 => neutral/undetected


@dataclass(frozen=True)
class MACDResult:
    macd: "pd.Series"
    signal: "pd.Series"
    hist: "pd.Series"


@dataclass(frozen=True)
class BollingerBands:
    middle: "pd.Series"
    upper: "pd.Series"
    lower: "pd.Series"
    bandwidth: "pd.Series"


def _require_deps() -> None:
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency. Install with: pip install pandas numpy"
        ) from e


def _close(df: "pd.DataFrame") -> "pd.Series":
    if "Close" not in df.columns:
        raise ValueError("Expected column 'Close'")
    return df["Close"]


def sma(series: "pd.Series", window: int) -> "pd.Series":
    """Simple moving average."""
    _require_deps()
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: "pd.Series", span: int) -> "pd.Series":
    """Exponential moving average (EWM, span parameterization)."""
    _require_deps()
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: "pd.Series", period: int = 14) -> "pd.Series":
    """
    Relative Strength Index (Wilder-style smoothing via EMA with alpha=1/period).
    Returns a series in [0, 100].
    """
    _require_deps()
    import numpy as np

    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    # Wilder's smoothing
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = roll_up / roll_down.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(
    series: "pd.Series",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """
    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    _require_deps()
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return MACDResult(macd=macd_line, signal=signal_line, hist=hist)


def bollinger_bands(
    series: "pd.Series",
    window: int = 20,
    num_std: float = 2.0,
) -> BollingerBands:
    """Bollinger Bands around SMA(window) with +/- num_std * rolling std."""
    _require_deps()
    import numpy as np

    mid = sma(series, window)
    sd = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)
    return BollingerBands(middle=mid, upper=upper, lower=lower, bandwidth=bandwidth)


def _find_pivots(
    series: "pd.Series",
    left: int = 3,
    right: int = 3,
    *,
    kind: str,
) -> List[Tuple[int, float]]:
    """
    Find local extrema using a fixed neighborhood.
    Returns list of (pos, value) where pos is integer positional index.

    kind: "high" or "low"
    """
    _require_deps()
    import numpy as np

    if kind not in ("high", "low"):
        raise ValueError("kind must be 'high' or 'low'")

    x = series.to_numpy(dtype=float)
    n = len(x)
    piv: List[Tuple[int, float]] = []
    if n == 0:
        return piv

    for i in range(left, n - right):
        window = x[i - left : i + right + 1]
        if np.any(np.isnan(window)):
            continue
        center = x[i]
        if kind == "high":
            if center == np.max(window) and np.sum(window == center) == 1:
                piv.append((i, float(center)))
        else:
            if center == np.min(window) and np.sum(window == center) == 1:
                piv.append((i, float(center)))
    return piv


def _pct_diff(a: float, b: float) -> float:
    denom = (abs(a) + abs(b)) / 2.0
    if denom == 0:
        return 0.0
    return abs(a - b) / denom


def _clip_score(x: float) -> float:
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return float(x)


def head_shoulders_score(
    df: "pd.DataFrame",
    *,
    lookback: int = 200,
    tolerance: float = 0.03,
    pivot_left: int = 3,
    pivot_right: int = 3,
) -> float:
    """
    Heuristic head & shoulders detection on Close (bearish) and inverse (bullish).
    Score is driven by neckline break direction.
    """
    _require_deps()
    import numpy as np

    close = _close(df).tail(lookback).dropna()
    if len(close) < 20:
        return 0.0

    highs = _find_pivots(close, left=pivot_left, right=pivot_right, kind="high")
    lows = _find_pivots(close, left=pivot_left, right=pivot_right, kind="low")
    if len(highs) < 3 or len(lows) < 2:
        return 0.0

    # --- bearish H&S: three pivot highs with middle highest; two lows define neckline
    score_bear = 0.0
    for (i1, h1), (i2, h2), (i3, h3) in zip(highs, highs[1:], highs[2:]):
        if not (i1 < i2 < i3):
            continue
        if not (h2 > h1 and h2 > h3):
            continue
        if _pct_diff(h1, h3) > tolerance:
            continue
        # find lowest lows between shoulders/head and head/shoulder
        lows_12 = [v for (j, v) in lows if i1 < j < i2]
        lows_23 = [v for (j, v) in lows if i2 < j < i3]
        if not lows_12 or not lows_23:
            continue
        neckline = (min(lows_12) + min(lows_23)) / 2.0
        last = float(close.iloc[-1])
        # break below neckline => bearish
        if last < neckline * (1.0 - tolerance):
            score_bear = min(-0.8, score_bear)  # strong bearish
        else:
            # partial/forming pattern
            score_bear = min(-0.3, score_bear)

    # --- inverse H&S: three pivot lows with middle lowest; highs define neckline
    score_bull = 0.0
    for (i1, l1), (i2, l2), (i3, l3) in zip(lows, lows[1:], lows[2:]):
        if not (i1 < i2 < i3):
            continue
        if not (l2 < l1 and l2 < l3):
            continue
        if _pct_diff(l1, l3) > tolerance:
            continue
        highs_12 = [v for (j, v) in highs if i1 < j < i2]
        highs_23 = [v for (j, v) in highs if i2 < j < i3]
        if not highs_12 or not highs_23:
            continue
        neckline = (max(highs_12) + max(highs_23)) / 2.0
        last = float(close.iloc[-1])
        if last > neckline * (1.0 + tolerance):
            score_bull = max(0.8, score_bull)
        else:
            score_bull = max(0.3, score_bull)

    # If both detected, prefer the one with stronger absolute evidence.
    if abs(score_bull) >= abs(score_bear):
        return _clip_score(score_bull)
    return _clip_score(score_bear)


def double_top_bottom_score(
    df: "pd.DataFrame",
    *,
    lookback: int = 200,
    tolerance: float = 0.02,
    pivot_left: int = 3,
    pivot_right: int = 3,
) -> float:
    """
    Double top (bearish) / double bottom (bullish) using pivot highs/lows.
    Score strengthens when the 'confirmation' level is broken.
    """
    _require_deps()

    close = _close(df).tail(lookback).dropna()
    if len(close) < 20:
        return 0.0

    highs = _find_pivots(close, left=pivot_left, right=pivot_right, kind="high")
    lows = _find_pivots(close, left=pivot_left, right=pivot_right, kind="low")
    if len(highs) < 2 and len(lows) < 2:
        return 0.0

    last = float(close.iloc[-1])
    best = 0.0

    # Double top: two similar highs with a valley low between -> bearish if last < valley
    for (i1, h1), (i2, h2) in zip(highs, highs[1:]):
        if _pct_diff(h1, h2) > tolerance:
            continue
        valley = [v for (j, v) in lows if i1 < j < i2]
        if not valley:
            continue
        lvl = min(valley)
        if last < lvl * (1.0 - tolerance):
            best = min(best, -0.8)
        else:
            best = min(best, -0.25)

    # Double bottom: two similar lows with a peak high between -> bullish if last > peak
    for (i1, l1), (i2, l2) in zip(lows, lows[1:]):
        if _pct_diff(l1, l2) > tolerance:
            continue
        peak = [v for (j, v) in highs if i1 < j < i2]
        if not peak:
            continue
        lvl = max(peak)
        if last > lvl * (1.0 + tolerance):
            best = max(best, 0.8)
        else:
            best = max(best, 0.25)

    return _clip_score(best)


def support_resistance_score(
    df: "pd.DataFrame",
    *,
    lookback: int = 200,
    pivot_left: int = 3,
    pivot_right: int = 3,
    cluster_tol: float = 0.01,
    proximity: float = 0.01,
) -> float:
    """
    Detect simple support/resistance zones by clustering pivot highs/lows.
    Score:
    - near support => bullish (+)
    - near resistance => bearish (-)
    - breakout above resistance => bullish (strong)
    - breakdown below support => bearish (strong)
    """
    _require_deps()
    import numpy as np

    close = _close(df).tail(lookback).dropna()
    if len(close) < 20:
        return 0.0

    highs = [v for _, v in _find_pivots(close, left=pivot_left, right=pivot_right, kind="high")]
    lows = [v for _, v in _find_pivots(close, left=pivot_left, right=pivot_right, kind="low")]
    if not highs and not lows:
        return 0.0

    def cluster(levels: Sequence[float]) -> List[float]:
        lv = sorted(float(x) for x in levels)
        out: List[float] = []
        for x in lv:
            placed = False
            for k in range(len(out)):
                if abs(x - out[k]) / max(abs(out[k]), 1e-12) <= cluster_tol:
                    out[k] = (out[k] + x) / 2.0
                    placed = True
                    break
            if not placed:
                out.append(x)
        return out

    res_levels = cluster(highs)
    sup_levels = cluster(lows)
    last = float(close.iloc[-1])

    # nearest levels
    nearest_res = min(res_levels, key=lambda x: abs(x - last)) if res_levels else None
    nearest_sup = min(sup_levels, key=lambda x: abs(x - last)) if sup_levels else None

    score = 0.0
    if nearest_res is not None:
        if last > nearest_res * (1.0 + proximity):
            score = max(score, 0.8)  # breakout
        elif abs(last - nearest_res) / nearest_res <= proximity:
            score = min(score, -0.35)  # at resistance

    if nearest_sup is not None:
        if last < nearest_sup * (1.0 - proximity):
            score = min(score, -0.8)  # breakdown
        elif abs(last - nearest_sup) / nearest_sup <= proximity:
            score = max(score, 0.35)  # at support

    # If both are close, dampen slightly (range-bound)
    if (
        nearest_res is not None
        and nearest_sup is not None
        and abs(last - nearest_res) / nearest_res <= proximity
        and abs(last - nearest_sup) / nearest_sup <= proximity
    ):
        score *= 0.5

    if np.isnan(score):
        return 0.0
    return _clip_score(score)


def _linear_fit_slope(y: "pd.Series") -> float:
    _require_deps()
    import numpy as np

    x = np.arange(len(y), dtype=float)
    yy = y.to_numpy(dtype=float)
    if len(yy) < 2 or np.any(np.isnan(yy)):
        return 0.0
    slope, _ = np.polyfit(x, yy, 1)
    return float(slope)


def triangle_score(
    df: "pd.DataFrame",
    *,
    lookback: int = 120,
    band_window: int = 20,
    breakout_tol: float = 0.002,
) -> float:
    """
    Triangle-ish consolidation:
    - rolling max slope < 0 and rolling min slope > 0 (converging),
      plus shrinking rolling range.
    Score based on breakout direction relative to last upper/lower bands.
    """
    _require_deps()
    import numpy as np
    import pandas as pd

    close = _close(df).tail(lookback).dropna()
    if len(close) < band_window + 5:
        return 0.0

    roll_max = close.rolling(band_window, min_periods=band_window).max()
    roll_min = close.rolling(band_window, min_periods=band_window).min()
    rng = (roll_max - roll_min) / close.rolling(band_window, min_periods=band_window).mean()

    tail_max = roll_max.dropna().tail(band_window)
    tail_min = roll_min.dropna().tail(band_window)
    tail_rng = rng.dropna().tail(band_window)
    if len(tail_max) < 5 or len(tail_min) < 5 or len(tail_rng) < 5:
        return 0.0

    slope_hi = _linear_fit_slope(tail_max)
    slope_lo = _linear_fit_slope(tail_min)
    slope_rng = _linear_fit_slope(tail_rng)

    converging = (slope_hi < 0.0) and (slope_lo > 0.0) and (slope_rng < 0.0)
    if not converging:
        return 0.0

    last = float(close.iloc[-1])
    upper = float(roll_max.iloc[-1])
    lower = float(roll_min.iloc[-1])

    if last > upper * (1.0 + breakout_tol):
        return 0.7
    if last < lower * (1.0 - breakout_tol):
        return -0.7

    # forming triangle: neutral-to-slight continuation bias (use last return sign)
    ret = float(close.pct_change().tail(5).mean())
    if np.isnan(ret):
        return 0.0
    return _clip_score(0.15 if ret >= 0 else -0.15)


def flag_score(
    df: "pd.DataFrame",
    *,
    lookback: int = 120,
    impulse_window: int = 20,
    flag_window: int = 20,
    impulse_min_return: float = 0.06,
    max_flag_slope_frac: float = 0.25,
) -> float:
    """
    Flag detection (very heuristic):
    - find strong impulse move in last (impulse_window) bars
    - followed by (flag_window) bars of tighter consolidation with mild counter-trend slope
    Score:
    - bullish flag after bullish impulse
    - bearish flag after bearish impulse
    """
    _require_deps()
    import numpy as np

    close = _close(df).tail(lookback).dropna()
    if len(close) < impulse_window + flag_window + 5:
        return 0.0

    impulse = close.iloc[-(impulse_window + flag_window) : -flag_window]
    flag = close.iloc[-flag_window:]

    imp_ret = float(impulse.iloc[-1] / impulse.iloc[0] - 1.0)
    if abs(imp_ret) < impulse_min_return:
        return 0.0

    # Flag should be "tighter" than impulse
    imp_vol = float(impulse.pct_change().std())
    flag_vol = float(flag.pct_change().std())
    if np.isnan(imp_vol) or np.isnan(flag_vol) or flag_vol > imp_vol:
        return 0.0

    slope_imp = _linear_fit_slope(impulse)
    slope_flag = _linear_fit_slope(flag)
    if slope_imp == 0.0:
        return 0.0

    # Counter-trend slope in the flag should be smaller magnitude than impulse slope
    if abs(slope_flag) > abs(slope_imp) * max_flag_slope_frac:
        return 0.0

    # Continuation bias: sign determined by impulse
    base = 0.45 if imp_ret > 0 else -0.45

    # If price breaks beyond flag range in impulse direction, strengthen
    last = float(close.iloc[-1])
    hi = float(flag.max())
    lo = float(flag.min())
    if imp_ret > 0 and last >= hi:
        return 0.75
    if imp_ret < 0 and last <= lo:
        return -0.75
    return _clip_score(base)


def scores(
    df: "pd.DataFrame",
    *,
    lookback: int = 200,
) -> Dict[str, float]:
    """Compute all pattern scores on the given OHLCV DataFrame."""
    return {
        "head_shoulders": head_shoulders_score(df, lookback=lookback),
        "double_top_bottom": double_top_bottom_score(df, lookback=lookback),
        "triangles": triangle_score(df, lookback=min(lookback, 160)),
        "flags": flag_score(df, lookback=min(lookback, 160)),
        "support_resistance": support_resistance_score(df, lookback=lookback),
    }


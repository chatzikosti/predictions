from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union


def _require_deps() -> None:
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency. Install with: pip install numpy pandas"
        ) from e


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency. Install with: pip install scikit-learn"
        ) from e


@dataclass(frozen=True)
class FeatureConfig:
    lookback: int = 260
    sma_fast: int = 10
    sma_slow: int = 50
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    bb_window: int = 20
    bb_num_std: float = 2.0


def build_features(
    ohlcv: "pd.DataFrame",
    *,
    sentiment_score: float = 0.0,
    cfg: FeatureConfig = FeatureConfig(),
) -> "pd.DataFrame":
    """
    Build a feature frame from OHLCV + a scalar sentiment_score.

    Expected OHLCV columns: Open, High, Low, Close, Volume
    Output index aligns with input; rows with insufficient history will be NaN.
    """
    _require_deps()
    import numpy as np
    import pandas as pd

    from utils.technical_analysis import (
        bollinger_bands,
        macd,
        rsi,
        scores as pattern_scores,
        sma,
        ema,
    )

    df = ohlcv.tail(cfg.lookback).copy()
    if "Close" not in df.columns:
        raise ValueError("OHLCV must contain column 'Close'")

    close = df["Close"].astype(float)

    # Returns / momentum
    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)

    # Trend
    sma_fast = sma(close, cfg.sma_fast)
    sma_slow = sma(close, cfg.sma_slow)
    ema_fast = ema(close, cfg.ema_fast)
    ema_slow = ema(close, cfg.ema_slow)

    trend_sma = (sma_fast / sma_slow) - 1.0
    trend_ema = (ema_fast / ema_slow) - 1.0

    # RSI scaled to roughly [-1, 1]
    rsi_v = rsi(close, period=cfg.rsi_period)
    rsi_scaled = (rsi_v - 50.0) / 50.0

    # MACD histogram (scaled by price)
    macd_res = macd(close, fast=cfg.ema_fast, slow=cfg.ema_slow, signal=9)
    macd_hist_scaled = macd_res.hist / close.replace(0.0, np.nan)

    # Bollinger %B (position inside the band)
    bb = bollinger_bands(close, window=cfg.bb_window, num_std=cfg.bb_num_std)
    bb_range = (bb.upper - bb.lower).replace(0.0, np.nan)
    bb_percent_b = (close - bb.lower) / bb_range  # ~[0,1] usually
    bb_bandwidth = bb.bandwidth

    # Volume signals
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)
    vol_z20 = (vol - vol.rolling(20, min_periods=20).mean()) / vol.rolling(20, min_periods=20).std(ddof=0)

    # Pattern scores: constant across the lookback window (computed using recent history)
    p = pattern_scores(df, lookback=min(cfg.lookback, len(df)))

    feats = pd.DataFrame(
        {
            "ret1": ret1,
            "ret5": ret5,
            "ret20": ret20,
            "trend_sma": trend_sma,
            "trend_ema": trend_ema,
            "rsi": rsi_scaled,
            "macd_hist": macd_hist_scaled,
            "bb_percent_b": bb_percent_b,
            "bb_bandwidth": bb_bandwidth,
            "vol_z20": vol_z20,
            "pat_head_shoulders": float(p["head_shoulders"]),
            "pat_double_top_bottom": float(p["double_top_bottom"]),
            "pat_triangles": float(p["triangles"]),
            "pat_flags": float(p["flags"]),
            "pat_support_resistance": float(p["support_resistance"]),
            "news_sentiment": float(sentiment_score),
        },
        index=df.index,
    )

    return feats


def make_dataset_next_day_direction(
    ohlcv: "pd.DataFrame",
    *,
    sentiment_score: float = 0.0,
    cfg: FeatureConfig = FeatureConfig(),
) -> Tuple["pd.DataFrame", "pd.Series"]:
    """
    Build X, y for next-day direction classification.

    y = 1 if next day's close return > 0 else 0
    Rows where y is not defined (last row) are dropped.
    """
    _require_deps()
    import numpy as np

    feats = build_features(ohlcv, sentiment_score=sentiment_score, cfg=cfg)
    close = ohlcv["Close"].astype(float).reindex(feats.index)
    next_ret = close.pct_change().shift(-1)
    y = (next_ret > 0).astype(int)

    # Drop rows with NaNs in features or undefined label
    mask = feats.notna().all(axis=1) & next_ret.notna()
    X = feats.loc[mask]
    y = y.loc[mask]
    return X, y


class NextDayMovementPredictor:
    """
    Simple next-day up/down predictor using scikit-learn logistic regression.

    The model predicts P(up tomorrow) given today's features.
    """

    def __init__(
        self,
        *,
        feature_cfg: FeatureConfig = FeatureConfig(),
        C: float = 1.0,
        max_iter: int = 2000,
    ) -> None:
        self.feature_cfg = feature_cfg
        self.C = float(C)
        self.max_iter = int(max_iter)
        self._model = None
        self._feature_columns: Optional[Sequence[str]] = None

    def fit(
        self,
        ohlcv: "pd.DataFrame",
        *,
        sentiment_score: float = 0.0,
    ) -> "NextDayMovementPredictor":
        _require_sklearn()
        import numpy as np

        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = make_dataset_next_day_direction(
            ohlcv, sentiment_score=sentiment_score, cfg=self.feature_cfg
        )
        if len(X) < 30:
            raise ValueError("Not enough training rows after feature construction (need >= 30).")

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")),
            ]
        )
        pipe.fit(X.values, y.values)
        self._model = pipe
        self._feature_columns = list(X.columns)
        return self

    def predict_proba_up(
        self,
        ohlcv: "pd.DataFrame",
        *,
        sentiment_score: float = 0.0,
    ) -> float:
        """
        Predict P(up next day) for the most recent row.
        """
        if self._model is None or self._feature_columns is None:
            raise ValueError("Model not fit yet. Call fit() first.")

        feats = build_features(ohlcv, sentiment_score=sentiment_score, cfg=self.feature_cfg)
        last = feats.iloc[[-1]][list(self._feature_columns)]
        proba = float(self._model.predict_proba(last.values)[0, 1])
        return proba

    def predict_direction(
        self,
        ohlcv: "pd.DataFrame",
        *,
        sentiment_score: float = 0.0,
        threshold: float = 0.5,
    ) -> int:
        """Return 1 for up, 0 for down/flat."""
        p = self.predict_proba_up(ohlcv, sentiment_score=sentiment_score)
        return int(p >= float(threshold))


def fit_predict_multi_ticker(
    ohlcv_by_ticker: Dict[str, "pd.DataFrame"],
    *,
    sentiment_by_ticker: Optional[Dict[str, float]] = None,
    model_kwargs: Optional[dict] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience helper to train and predict per ticker.

    Returns dict:
      {ticker: {"p_up": float, "direction": 0|1}}
    """
    sentiment_by_ticker = sentiment_by_ticker or {}
    model_kwargs = model_kwargs or {}
    out: Dict[str, Dict[str, float]] = {}
    for t, df in ohlcv_by_ticker.items():
        sent = float(sentiment_by_ticker.get(t, 0.0))
        m = NextDayMovementPredictor(**model_kwargs).fit(df, sentiment_score=sent)
        p = m.predict_proba_up(df, sentiment_score=sent)
        out[t] = {"p_up": float(p), "direction": int(p >= 0.5)}
    return out


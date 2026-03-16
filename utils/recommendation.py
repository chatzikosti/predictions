from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Prediction:
    """
    Next-day movement prediction for a ticker.

    p_up is the model probability that price closes up tomorrow.
    """

    ticker: str
    p_up: float


@dataclass(frozen=True)
class RiskMetrics:
    """
    Risk-related inputs used for recommendations.

    - volatility: typical daily volatility (e.g. std of daily returns). Higher => riskier.
    - trend_confidence: [0, 1] confidence in the trend signal (higher => better).
    """

    volatility: float
    trend_confidence: float


@dataclass(frozen=True)
class Recommendation:
    ticker: str
    action: str  # "buy" | "sell" | "hold"
    score: float  # higher is better (risk-adjusted)
    risk: float  # lower is better
    explanation: str


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _to_prediction_map(
    predicted: Union[
        Sequence[Prediction],
        Mapping[str, float],
        Mapping[str, Mapping[str, float]],
    ]
) -> Dict[str, Prediction]:
    """
    Accepts any of:
    - [Prediction(...), ...]
    - {"AAPL": 0.62, "MSFT": 0.48} (values are p_up)
    - {"AAPL": {"p_up": 0.62}, ...} (e.g. output from models/predictor helpers)
    """
    out: Dict[str, Prediction] = {}
    if isinstance(predicted, Mapping):
        for t, v in predicted.items():
            if isinstance(v, Mapping):
                p = float(v.get("p_up", 0.5))
            else:
                p = float(v)
            out[str(t)] = Prediction(ticker=str(t), p_up=_clip01(p))
        return out

    for p in predicted:
        out[str(p.ticker)] = Prediction(ticker=str(p.ticker), p_up=_clip01(float(p.p_up)))
    return out


def _normalize_risk(vol: float, vol_floor: float = 0.0) -> float:
    """
    Normalize volatility to a soft [0, 1] scale.
    Interprets vol as typical daily std of returns (e.g. 0.02 = 2%).

    The mapping is intentionally simple:
    - <= 1% daily vol => low risk (~0.2)
    - ~3% daily vol => medium risk (~0.6)
    - >= 6% daily vol => high risk (~0.85+)
    """
    v = max(float(vol), float(vol_floor))
    # logistic-ish squash without extra deps
    # pivot at 3% and scale so 6% is much riskier
    pivot = 0.03
    scale = 0.02
    x = (v - pivot) / scale
    # sigmoid
    import math

    s = 1.0 / (1.0 + math.exp(-x))
    return float(0.1 + 0.9 * s)


def _action_from_p_up(p_up: float, threshold: float) -> str:
    if p_up >= threshold:
        return "buy"
    if p_up <= 1.0 - threshold:
        return "sell"
    return "hold"


def recommend_positions(
    predicted_movements: Union[
        Sequence[Prediction],
        Mapping[str, float],
        Mapping[str, Mapping[str, float]],
    ],
    risk: Mapping[str, RiskMetrics],
    *,
    top_k: int = 3,
    threshold: float = 0.57,
    risk_aversion: float = 0.65,
) -> List[Recommendation]:
    """
    Recommend top positions (buy/sell/hold) with minimum risk.

    Inputs:
    - predicted_movements: per-ticker p_up values or compatible dict structure
    - risk: per-ticker RiskMetrics(volatility, trend_confidence)

    Output:
    - list of up to top_k Recommendation objects, ranked by risk-adjusted score

    Scoring:
    - expected edge uses distance from 0.5: edge = |p_up - 0.5| * 2 in [0,1]
    - higher trend_confidence increases score
    - higher volatility increases risk penalty
    """
    preds = _to_prediction_map(predicted_movements)
    if top_k <= 0:
        return []

    recs: List[Recommendation] = []
    for t, pred in preds.items():
        if t not in risk:
            # If risk isn't supplied for a ticker, skip it so "minimum risk" is honored.
            continue

        r = risk[t]
        p_up = _clip01(pred.p_up)
        action = _action_from_p_up(p_up, float(threshold))

        edge = abs(p_up - 0.5) * 2.0  # 0..1
        tc = _clip01(float(r.trend_confidence))
        risk_norm = _normalize_risk(float(r.volatility))

        # Prefer low risk and high-confidence edge. Holds are allowed but naturally score lower
        # unless risk is very low.
        base = edge * (0.5 + 0.5 * tc)
        hold_penalty = 0.15 if action == "hold" else 0.0
        score = base - float(risk_aversion) * risk_norm - hold_penalty

        direction = "up" if p_up >= 0.5 else "down"
        conf_pct = round(max(p_up, 1.0 - p_up) * 100.0, 1)
        edge_pct = round(edge * 100.0, 1)
        vol_pct = round(float(r.volatility) * 100.0, 2)
        tc_pct = round(tc * 100.0, 0)

        why_parts = [
            f"Model leans {direction} (p_up={p_up:.2f}, edge≈{edge_pct}%).",
            f"Volatility≈{vol_pct}%/day (risk={risk_norm:.2f}).",
            f"Trend confidence≈{int(tc_pct)}%.",
        ]
        if action == "buy":
            why_parts.append("Suggested BUY because upside probability is meaningfully above 50%.")
        elif action == "sell":
            why_parts.append("Suggested SELL because downside probability is meaningfully above 50%.")
        else:
            why_parts.append("Suggested HOLD because the signal is not strong enough after risk adjustment.")

        recs.append(
            Recommendation(
                ticker=t,
                action=action,
                score=float(score),
                risk=float(risk_norm),
                explanation=" ".join(why_parts),
            )
        )

    # Rank by: lowest risk preferred, then highest score, then strongest edge
    recs.sort(key=lambda x: (x.risk, -x.score, -abs(preds[x.ticker].p_up - 0.5)))
    return recs[: min(int(top_k), len(recs))]


def recommend_top3(
    predicted_movements: Union[
        Sequence[Prediction],
        Mapping[str, float],
        Mapping[str, Mapping[str, float]],
    ],
    risk: Mapping[str, RiskMetrics],
    *,
    threshold: float = 0.57,
    risk_aversion: float = 0.65,
) -> List[Recommendation]:
    """Convenience wrapper for the common 'top 3' request."""
    return recommend_positions(
        predicted_movements,
        risk,
        top_k=3,
        threshold=threshold,
        risk_aversion=risk_aversion,
    )


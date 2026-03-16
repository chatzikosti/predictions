import streamlit as st
import pandas as pd

from app import get_predictions


st.set_page_config(page_title="Stock Predictions", layout="wide")

st.title("Stock Predictions")
st.caption(
    "Daily stock insights combining price action, technical patterns, and news sentiment "
    "to suggest simple BUY / HOLD / SELL ideas."
)

with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers = st.text_input(
            "Tickers",
            "AAPL,MSFT,NVDA,TSLA,AMZN",
            help="Enter one or more ticker symbols, separated by commas.",
        )
    with col2:
        period = st.selectbox(
            "History window",
            ["1mo", "3mo", "6mo", "1y"],
            index=2,
            help="How far back to look when building indicators.",
        )

run = st.button("Run predictions", type="primary")


def _risk_label(volatility: float) -> str:
    v = volatility * 100.0
    if v < 1.0:
        return "Very low risk"
    if v < 2.0:
        return "Low risk"
    if v < 3.5:
        return "Moderate risk"
    if v < 6.0:
        return "High risk"
    return "Very high risk"


def _confidence_label(trend_conf: float) -> str:
    c = trend_conf
    if c < 0.25:
        return "Very low confidence"
    if c < 0.5:
        return "Low confidence"
    if c < 0.7:
        return "Medium confidence"
    if c < 0.9:
        return "High confidence"
    return "Very high confidence"


def _badge_style(action: str) -> str:
    a = action.upper()
    if a == "BUY":
        color = "#16a34a"  # green
    elif a == "SELL":
        color = "#dc2626"  # red
    else:
        color = "#eab308"  # amber for HOLD
    return (
        f"background-color:{color};color:white;padding:0.15rem 0.6rem;"
        "border-radius:999px;font-weight:600;font-size:0.9rem;"
    )


if run:
    with st.spinner("Running analysis and building recommendations…"):
        try:
            results = get_predictions(tickers=tickers, period=period)
        except Exception as e:
            st.error("Something went wrong while fetching predictions. "
                     "Please check your tickers and try again.")
            st.caption(str(e))
        else:
            tickers_list = results["tickers"]
            tech = results["technical_summary"]
            sentiment = results["sentiment_by_ticker"]
            preds = results["predictions"]
            risk = results["risk_by_ticker"]
            recs = results["recommendations"]

            # --- Summary table ranked by recommendation strength
            st.subheader("Summary ranking")
            if recs:
                summary_rows = []
                for r in recs:
                    t = r["ticker"]
                    summary_rows.append(
                        {
                            "Ticker": t,
                            "Action": r["action"].upper(),
                            "Score": r["score"],
                            "Risk": r["risk"],
                            "Signal (p_up)": preds[t]["p_up"],
                        }
                    )
                summary_df = (
                    pd.DataFrame(summary_rows)
                    .set_index("Ticker")
                    .sort_values("Score", ascending=False)
                )
                st.dataframe(
                    summary_df.round(
                        {
                            "Score": 3,
                            "Risk": 2,
                            "Signal (p_up)": 3,
                        }
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No recommendations are available for the selected inputs.")

            # --- Per‑stock cards
            st.subheader("Details per stock")
            for t in tickers_list:
                if t not in tech:
                    continue
                ts = tech[t]
                p_up = float(preds[t]["p_up"])
                sent = float(sentiment.get(t, 0.0))
                r = risk[t]

                # Find recommendation entry for this ticker
                rec = next((x for x in recs if x["ticker"] == t), None)
                action = rec["action"].upper() if rec else "HOLD"
                score = rec["score"] if rec else 0.0

                badge_html = f'<span style="{_badge_style(action)}">{action}</span>'

                with st.expander(f"{t} – recommendation and indicators", expanded=True):
                    top_cols = st.columns([2, 2, 2])
                    with top_cols[0]:
                        st.markdown("**Current price**")
                        st.metric(
                            label="",
                            value=f"${ts['close']:.2f}",
                            delta=None,
                        )
                    with top_cols[1]:
                        st.markdown("**Recommendation**")
                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.caption(f"Model score: {score:+.3f}")
                    with top_cols[2]:
                        st.markdown("**Probability up**")
                        st.metric(
                            label="p(up tomorrow)",
                            value=f"{p_up*100:.1f}%",
                        )
                        st.caption(f"News sentiment: {sent:+.2f}")

                    mid_cols = st.columns(2)
                    with mid_cols[0]:
                        st.markdown("**Risk profile**")
                        vol = float(r["volatility"])
                        tc = float(r["trend_confidence"])
                        st.write(_risk_label(vol))
                        st.write(_confidence_label(tc))
                    with mid_cols[1]:
                        st.markdown("**Key technical indicators**")
                        ind_df = pd.DataFrame(
                            [
                                {
                                    "RSI (14)": ts["rsi14"],
                                    "SMA (10)": ts["sma10"],
                                    "SMA (50)": ts["sma50"],
                                    "Bollinger %B": ts["bb_percent_b"],
                                }
                            ],
                            index=[t],
                        )
                        st.dataframe(ind_df.round(2), use_container_width=True)

                    st.markdown("**Pattern scores**")
                    patt_df = pd.DataFrame(
                        [
                            {
                                "Head & Shoulders": ts["pat_head_shoulders"],
                                "Double Top/Bottom": ts["pat_double_top_bottom"],
                                "Triangles": ts["pat_triangles"],
                                "Flags": ts["pat_flags"],
                                "Support/Resistance": ts["pat_support_resistance"],
                            }
                        ],
                        index=[t],
                    )
                    st.dataframe(patt_df.round(2), use_container_width=True)


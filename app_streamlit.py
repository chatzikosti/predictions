import streamlit as st
import pandas as pd

from app import scan_intraday_opportunities


st.set_page_config(page_title="Intraday Trading Assistant", layout="wide")

st.title("Intraday Trading Assistant")
st.caption(
    "Scans the S&P 500 each morning and highlights 5–10 intraday trade ideas using "
    "trend, momentum, volume, and breakout/pattern context."
)


def _badge(action: str) -> str:
    a = (action or "AVOID").upper()
    if a == "BUY":
        color = "#16a34a"
    elif a == "SELL SHORT":
        color = "#dc2626"
    else:
        color = "#6b7280"  # gray
    return (
        f"<span style='background:{color};color:white;padding:0.2rem 0.65rem;"
        "border-radius:999px;font-weight:700;font-size:0.85rem;'>"
        f"{a}</span>"
    )


def _confidence_color(conf: str) -> str:
    c = (conf or "").lower()
    if c == "high":
        return "#16a34a"
    if c == "medium":
        return "#eab308"
    return "#6b7280"


def _risk_english(risk: float) -> str:
    if risk < 0.45:
        return "Low risk"
    if risk < 0.7:
        return "Medium risk"
    return "High risk"


def _conf_english(conf: str) -> str:
    c = (conf or "").lower()
    if c == "high":
        return "High confidence"
    if c == "medium":
        return "Medium confidence"
    return "Low confidence"


top = st.container()
with top:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        capital = st.number_input(
            "Total capital ($)",
            min_value=100.0,
            value=10000.0,
            step=100.0,
            help="Position size is calculated as risking 1% of this amount per trade.",
        )
    with c2:
        top_n = st.selectbox("Ideas to show", [5, 8, 10], index=1)
    with c3:
        scan = st.button("Scan now", type="primary", use_container_width=True)

st.divider()


if scan:
    with st.spinner("Scanning the S&P 500 (daily prefilter → intraday analysis)…"):
        try:
            results = scan_intraday_opportunities(capital=float(capital), top_n=int(top_n))
        except Exception as e:
            st.error("Scan failed. This can happen if data sources are unavailable.")
            st.caption(str(e))
        else:
            last_updated = results.get("last_updated_utc", "")
            universe_note = results.get("universe_note", "")
            opps = results.get("opportunities", []) or []
            macro = results.get("macro", {}) or {}

            header_cols = st.columns([2, 1])
            with header_cols[0]:
                st.subheader("Today’s opportunities")
                if universe_note:
                    st.caption(universe_note)
            with header_cols[1]:
                st.markdown("**Last updated (UTC)**")
                st.write(last_updated or "—")

            if not opps:
                st.info("No opportunities found that meet today’s alignment + risk/reward rules.")
            else:
                # --- Summary table
                st.subheader("Ranked summary")
                summary_rows = []
                for o in opps:
                    summary_rows.append(
                        {
                            "Ticker": o["ticker"],
                            "Action": o["action"],
                            "Confidence": o["confidence"],
                            "R/R": o["risk_reward"],
                            "Score": o["score"],
                            "Risk": o["risk"],
                            "Signal": o["signal"],
                        }
                    )
                summary_df = pd.DataFrame(summary_rows)
                conf_rank = {"High": 3, "Medium": 2, "Low": 1}
                summary_df["conf_rank"] = summary_df["Confidence"].map(conf_rank).fillna(0)
                summary_df = summary_df.sort_values(
                    ["conf_rank", "R/R", "Score"], ascending=[False, False, False]
                ).drop(columns=["conf_rank"])
                st.dataframe(
                    summary_df.round({"R/R": 2, "Score": 3, "Risk": 2, "Signal": 2}),
                    use_container_width=True,
                )

                st.subheader("Trade cards")
                for o in opps:
                    t = o["ticker"]
                    action = o["action"]
                    conf = o.get("confidence", "Low")

                    with st.container(border=True):
                        top_cols = st.columns([2, 2, 2, 2])
                        with top_cols[0]:
                            st.markdown(f"### {t}")
                            st.caption(f"Current price: ${o['current_price']:.2f}")
                        with top_cols[1]:
                            st.markdown("**Action**")
                            st.markdown(_badge(action), unsafe_allow_html=True)
                            st.caption(_risk_english(float(o["risk"])))
                        with top_cols[2]:
                            st.markdown("**Plan**")
                            st.metric("Entry", f"${o['entry']:.2f}")
                            st.metric("Target", f"${o['target']:.2f}")
                            st.metric("Stop", f"${o['stop']:.2f}")
                        with top_cols[3]:
                            st.markdown("**Sizing**")
                            st.metric("R/R", f"{o['risk_reward']:.1f}:1")
                            st.metric("Position size", f"{int(o['position_size'])} shares")
                            st.markdown(
                                f"<span style='color:{_confidence_color(conf)};font-weight:700;'>"
                                f"{_conf_english(conf)}</span>",
                                unsafe_allow_html=True,
                            )

                        st.caption(f"Best entry window: {o.get('best_entry_window','—')}")
                        st.write(o.get("explanation", ""))

            st.divider()
            st.subheader("Macro / news context")
            st.write(macro.get("note", "—"))
            headlines = macro.get("headlines", []) or []
            if headlines:
                with st.expander("Recent broad-market headlines"):
                    for h in headlines[:10]:
                        st.write(f"- {h}")


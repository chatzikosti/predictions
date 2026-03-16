import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from app import scan_intraday_opportunities


st.set_page_config(page_title="Intraday Trading Assistant", layout="wide")

st.title("Intraday Trading Assistant")
st.caption(
    "Scans the S&P 500 and key futures (Gold, Oil) each morning and highlights "
    "5–10 intraday trade ideas using trend, momentum, volume, and breakout/pattern context."
)

with st.expander("How to read this — click to open"):
    st.markdown(
        "- **BUY** — the analysis suggests the price is likely to go up. Consider opening a long position.\n"
        "- **SELL SHORT** — the analysis suggests the price is likely to go down. Consider opening a short position.\n"
        "- **AVOID** — signals are mixed or weak. No clear opportunity today.\n"
        "- **Entry price** — the price level at which to open your trade.\n"
        "- **Target price** — the price level at which to close and take your profit.\n"
        "- **Stop loss** — the price level at which to close and accept a small loss, to protect your capital.\n"
        "- **Risk/reward ratio** — how much you could gain vs how much you risk. 2:1 means you could gain $2 for every $1 risked. Always aim for 2:1 or better.\n"
        "- **Position size** — how many shares/units to buy based on your capital and 1% risk rule.\n"
        "- **Confidence** — how many signals agree. High = 4+ signals aligned, Medium = 3, Low = fewer than 3.\n"
        "- **Entry window** — the best time of day to place the trade based on intraday patterns.\n"
        "- **EMA 9 / EMA 21** — short term moving averages that show trend direction.\n"
        "- **RSI** — momentum indicator. Above 70 = overbought, below 30 = oversold.\n"
        "- **MACD** — shows momentum shifts. A crossover is often a trade signal.\n"
        "- **Volume confirmation** — a move on high volume is more reliable than one on low volume."
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


def _friendly_name(ticker: str) -> str:
    if ticker == "GC=F":
        return "Gold (Futures)"
    if ticker == "CL=F":
        return "WTI Crude Oil (Futures)"
    return ticker


def _load_intraday_15m(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="1d",
        interval="15m",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No intraday data available for {ticker}.")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def _add_ema(df: pd.DataFrame, span: int) -> pd.Series:
    return df["Close"].ewm(span=span, adjust=False, min_periods=span).mean()


def render_chart_for_opportunity(o: dict) -> None:
    ticker = o["ticker"]
    label = _friendly_name(ticker)

    try:
        df = _load_intraday_15m(ticker)
    except Exception as e:
        st.warning(f"Could not load intraday chart for {label}: {e}")
        return

    # Guard for empty / invalid intraday data (e.g. outside market hours)
    if df is None or df.empty or df["High"].dropna().empty:
        st.warning(
            f"No intraday chart data available for {label} right now. "
            "Try again during market hours (2:30pm–9pm UK time)."
        )
        return

    df = df.tail(120)  # show roughly the recent part of today

    ema9 = _add_ema(df, 9)
    ema21 = _add_ema(df, 21)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ema9,
            mode="lines",
            line=dict(color="#93c5fd", width=1.5),
            name="EMA 9",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ema21,
            mode="lines",
            line=dict(color="#f97316", width=1.5),
            name="EMA 21",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color="#4b5563",
            name="Volume",
        ),
        row=2,
        col=1,
    )

    x0 = df.index.min()
    x1 = df.index.max()
    entry = float(o["entry"])
    target = float(o["target"])
    stop = float(o["stop"])
    rr = float(o["risk_reward"])
    size = int(o["position_size"])

    # Horizontal lines
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=entry,
        y1=entry,
        xref="x",
        yref="y1",
        line=dict(color="#22c55e", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=x1,
        y=entry,
        xref="x",
        yref="y1",
        text=f"Entry ${entry:.2f}",
        showarrow=False,
        font=dict(color="#bbf7d0", size=11),
        xanchor="right",
    )

    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=target,
        y1=target,
        xref="x",
        yref="y1",
        line=dict(color="#3b82f6", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=x1,
        y=target,
        xref="x",
        yref="y1",
        text=f"Target ${target:.2f}",
        showarrow=False,
        font=dict(color="#bfdbfe", size=11),
        xanchor="right",
    )

    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=stop,
        y1=stop,
        xref="x",
        yref="y1",
        line=dict(color="#f97373", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=x1,
        y=stop,
        xref="x",
        yref="y1",
        text=f"Stop ${stop:.2f}",
        showarrow=False,
        font=dict(color="#fecaca", size=11),
        xanchor="right",
    )

    # Profit/risk shaded zones
    if target > entry:
        profit_y0, profit_y1 = entry, target
        risk_y0, risk_y1 = stop, entry
    else:
        profit_y0, profit_y1 = target, entry
        risk_y0, risk_y1 = entry, stop

    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=profit_y0,
        y1=profit_y1,
        xref="x",
        yref="y1",
        fillcolor="rgba(34,197,94,0.15)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=risk_y0,
        y1=risk_y1,
        xref="x",
        yref="y1",
        fillcolor="rgba(239,68,68,0.18)",
        line=dict(width=0),
        layer="below",
    )

    # Info box annotation
    info_text = (
        f"Entry: ${entry:.2f}<br>"
        f"Target: ${target:.2f}<br>"
        f"Stop: ${stop:.2f}<br>"
        f"R/R: {rr:.1f}:1<br>"
        f"Size: {size} shares"
    )
    fig.add_annotation(
        x=x0,
        y=max(
            float(df["High"].max()),
            float(target),
            float(entry),
            float(stop),
        ),
        xref="x",
        yref="y1",
        text=info_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(17,24,39,0.85)",
        bordercolor="#4b5563",
        borderwidth=1,
        font=dict(color="white", size=11),
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1f2937", row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


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

                        with st.expander("View chart (interactive intraday view)"):
                            render_chart_for_opportunity(o)

            st.divider()
            st.subheader("Macro / news context")
            st.write(macro.get("note", "—"))
            headlines = macro.get("headlines", []) or []
            if headlines:
                with st.expander("Recent broad-market headlines"):
                    for h in headlines[:10]:
                        st.write(f"- {h}")


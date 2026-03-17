import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from app import scan_intraday_opportunities

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python <3.9 fallback
    ZoneInfo = None  # type: ignore


st.set_page_config(page_title="Intraday Trading Assistant", layout="wide")

JOURNAL_PATH = "trade_journal.csv"
TICKER_BAR = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "GC=F", "CL=F", "SPY", "QQQ"]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_uk() -> datetime:
    if ZoneInfo is None:
        return _now_utc()
    return _now_utc().astimezone(ZoneInfo("Europe/London"))


def _in_market_hours_uk(ts: datetime) -> bool:
    t = ts.time()
    # London open 08:00–10:30, NY open 13:30–16:00, NY PM 16:00–20:00 (UK time)
    return (
        (t >= ts.replace(hour=8, minute=0, second=0, microsecond=0).time()
         and t <= ts.replace(hour=10, minute=30, second=0, microsecond=0).time())
        or (t >= ts.replace(hour=13, minute=30, second=0, microsecond=0).time()
            and t <= ts.replace(hour=16, minute=0, second=0, microsecond=0).time())
        or (t >= ts.replace(hour=16, minute=0, second=0, microsecond=0).time()
            and t <= ts.replace(hour=20, minute=0, second=0, microsecond=0).time())
    )


def _next_session_start_uk(ts: datetime) -> datetime:
    # Simple approximation: next is today 08:00, 13:30 or 16:00, or next day 08:00
    sessions = [
        ts.replace(hour=8, minute=0, second=0, microsecond=0),
        ts.replace(hour=13, minute=30, second=0, microsecond=0),
        ts.replace(hour=16, minute=0, second=0, microsecond=0),
    ]
    future = [s for s in sessions if s > ts]
    if future:
        return future[0]
    # next day London open
    return (ts + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)


def _load_journal() -> pd.DataFrame:
    if os.path.exists(JOURNAL_PATH):
        try:
            return pd.read_csv(JOURNAL_PATH, parse_dates=["timestamp"])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_journal(df: pd.DataFrame) -> None:
    try:
        df.to_csv(JOURNAL_PATH, index=False)
    except Exception:
        pass


def _ensure_session_state():
    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = None
    if "last_scan_time" not in st.session_state:
        st.session_state["last_scan_time"] = None
    if "next_scan_time" not in st.session_state:
        st.session_state["next_scan_time"] = None
    if "journal" not in st.session_state:
        st.session_state["journal"] = _load_journal()
    if "dobby_chat" not in st.session_state:
        st.session_state["dobby_chat"] = []
    if "ticker_cache" not in st.session_state:
        st.session_state["ticker_cache"] = {"time": None, "data": []}


_ensure_session_state()


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
        color = "#4b5563"
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


def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level columns that yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _scalar(val) -> float:
    """Safely convert a pandas scalar/Series/array to a Python float."""
    if hasattr(val, "item"):
        return float(val.item())
    if hasattr(val, "iloc"):
        return float(val.iloc[0])
    return float(val)


def _fetch_ticker_bar() -> list[dict]:
    cache = st.session_state["ticker_cache"]
    now = _now_utc()
    if cache["time"] is not None and (now - cache["time"]).total_seconds() < 60:
        return cache["data"]

    rows: list[dict] = []
    try:
        df = yf.download(
            TICKER_BAR,
            period="2d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        df = _flatten_df(df)
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance returns multi-index with first level OHLCV
            close = df["Close"]
        else:
            close = df
        for t in TICKER_BAR:
            try:
                series = close[t].dropna()
                if len(series) < 2:
                    price = float(series.iloc[-1])
                    pct = 0.0
                else:
                    prev, last = float(series.iloc[-2]), float(series.iloc[-1])
                    price = last
                    pct = (last / prev - 1.0) * 100.0
                rows.append({"ticker": t, "price": price, "pct": pct})
            except Exception:
                continue
    except Exception:
        pass

    cache["time"] = now
    cache["data"] = rows
    return rows


def render_ticker_bar() -> None:
    data = _fetch_ticker_bar()
    if not data:
        return
    with st.container():
        cols = st.columns(len(data))
        for col, row in zip(cols, data):
            color = "#16a34a" if row["pct"] >= 0 else "#dc2626"
            with col:
                st.markdown(
                    f"<div style='background:#000000;padding:0.25rem 0.5rem;"
                    f"border-radius:999px;text-align:center;border:1px solid #1e293b;'>"
                    f"<span style='color:#f1f5f9;font-weight:600;margin-right:0.35rem;'>{row['ticker']}</span>"
                    f"<span style='color:{color};font-weight:600;'>"
                    f"{row['price']:.2f} ({row['pct']:+.2f}%)"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )


render_ticker_bar()


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
    df = _flatten_df(df.copy())
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

    high_vals = df["High"].dropna()
    if df.empty or high_vals.empty:
        st.warning(
            f"No intraday chart data available for {label} right now. "
            "Try again during market hours (2:30pm–9pm UK time)."
        )
        return

    df = df.tail(120)
    high_vals = df["High"].dropna()

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

    # Safely get high/low scalars
    high_max = _scalar(high_vals.max())
    low_min = _scalar(df["Low"].dropna().min())

    for y_val, label_text, color, font_color in [
        (entry, f"Entry ${entry:.2f}", "#22c55e", "#bbf7d0"),
        (target, f"Target ${target:.2f}", "#3b82f6", "#bfdbfe"),
        (stop, f"Stop ${stop:.2f}", "#f97373", "#fecaca"),
    ]:
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=y_val, y1=y_val,
            xref="x", yref="y1",
            line=dict(color=color, width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=x1, y=y_val, xref="x", yref="y1",
            text=label_text, showarrow=False,
            font=dict(color=font_color, size=11),
            xanchor="right",
        )

    if target > entry:
        profit_y0, profit_y1 = entry, target
        risk_y0, risk_y1 = stop, entry
    else:
        profit_y0, profit_y1 = target, entry
        risk_y0, risk_y1 = entry, stop

    fig.add_shape(
        type="rect", x0=x0, x1=x1, y0=profit_y0, y1=profit_y1,
        xref="x", yref="y1",
        fillcolor="rgba(34,197,94,0.15)", line=dict(width=0), layer="below",
    )
    fig.add_shape(
        type="rect", x0=x0, x1=x1, y0=risk_y0, y1=risk_y1,
        xref="x", yref="y1",
        fillcolor="rgba(239,68,68,0.18)", line=dict(width=0), layer="below",
    )

    info_text = (
        f"Entry: ${entry:.2f}<br>"
        f"Target: ${target:.2f}<br>"
        f"Stop: ${stop:.2f}<br>"
        f"R/R: {rr:.1f}:1<br>"
        f"Size: {size} shares"
    )
    annotation_y = max(high_max, target, entry, stop)
    fig.add_annotation(
        x=x0, y=annotation_y,
        xref="x", yref="y1",
        text=info_text, showarrow=False, align="left",
        bgcolor="rgba(17,24,39,0.85)", bordercolor="#4b5563",
        borderwidth=1, font=dict(color="white", size=11),
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(color="#f1f5f9"),
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
            min_value=10.0,
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
                st.subheader("Today's opportunities")
                if universe_note:
                    st.caption(universe_note)
            with header_cols[1]:
                st.markdown("**Last updated (UTC)**")
                st.write(last_updated or "—")

            if not opps:
                st.info("No opportunities found that meet today's alignment + risk/reward rules.")
            else:
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
                            st.markdown(f"### {_friendly_name(t)}")
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

                        st.caption(f"Best entry window: {o.get('best_entry_window', '—')}")
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

# === Market Overview: Sector heatmap ===
st.divider()
with st.expander("Market Overview — sector heatmap"):
    sector_etfs = {
        "XLY": "Cons. Discretionary",
        "XLP": "Cons. Staples",
        "XLE": "Energy",
        "XLF": "Financials",
        "XLV": "Health Care",
        "XLI": "Industrials",
        "XLK": "Technology",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLB": "Materials",
        "XLC": "Communication",
    }
    try:
        data = yf.download(
            list(sector_etfs.keys()),
            period="2d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        data = _flatten_df(data)
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        changes = []
        labels = []
        for sym, name in sector_etfs.items():
            try:
                s = close[sym].dropna()
                if len(s) < 2:
                    continue
                prev, last = float(s.iloc[-2]), float(s.iloc[-1])
                pct = (last / prev - 1.0) * 100.0
                changes.append(pct)
                labels.append(name)
            except Exception:
                continue
        if changes:
            z = np.array([changes])
            fig_heat = px.imshow(
                z,
                x=labels,
                y=["Today"],
                color_continuous_scale=["#7f1d1d", "#f97316", "#16a34a"],
                aspect="auto",
            )
            fig_heat.update_layout(
                template="plotly_dark",
                coloraxis_showscale=False,
                paper_bgcolor="#0a0a0a",
                plot_bgcolor="#0a0a0a",
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Sector data is not available right now.")
    except Exception as e:
        st.info(f"Unable to load sector heatmap: {e}")

# === Trade Journal & Weekly P&L Summary ===
st.divider()
st.subheader("Trade Journal & Weekly Summary")

journal_df = _load_journal()
if journal_df.empty:
    st.info("No journal entries yet. Run a scan and record trade outcomes to build your history.")
else:
    st.markdown("**Recent journal entries**")
    show_cols = [
        "timestamp",
        "ticker",
        "action",
        "entry",
        "target",
        "stop",
        "rr",
        "position_size",
        "confidence",
        "frameworks",
        "timeframes",
        "outcome",
        "pnl_dollars",
    ]
    view_df = journal_df.copy()
    for col in show_cols:
        if col not in view_df.columns:
            view_df[col] = np.nan
    st.dataframe(
        view_df[show_cols].sort_values("timestamp", ascending=False).head(50),
        use_container_width=True,
    )

    st.markdown("**Weekly P&L summary**")
    today = _now_utc().date()
    start_week = today - timedelta(days=today.weekday())
    week_mask = journal_df["timestamp"].dt.date >= start_week
    week_df = journal_df[week_mask]
    taken = week_df[week_df["outcome"].isin(["Won", "Lost"])]
    total_signals = len(week_df)
    taken_count = len(taken)
    skipped_count = (week_df["outcome"] == "Skipped").sum()
    winners = (taken["pnl_dollars"] > 0).sum()
    losers = (taken["pnl_dollars"] < 0).sum()
    total_pnl = float(taken["pnl_dollars"].sum()) if taken_count else 0.0
    win_rate = (winners / taken_count * 100.0) if taken_count else 0.0
    avg_rr = float(taken["rr"].mean()) if "rr" in taken.columns and taken_count else 0.0

    cols = st.columns(4)
    cols[0].metric("Total signals (week)", total_signals)
    cols[1].metric("Taken / Skipped", f"{taken_count} / {skipped_count}")
    cols[2].metric("Win rate", f"{win_rate:.1f}%")
    cols[3].metric("Total P&L ($)", f"{total_pnl:+.2f}")

    best_trade = taken.loc[taken["pnl_dollars"].idxmax()] if not taken.empty else None
    worst_trade = taken.loc[taken["pnl_dollars"].idxmin()] if not taken.empty else None

    if best_trade is not None:
        st.write(
            f"**Best trade:** {best_trade['ticker']} "
            f"({best_trade['pnl_dollars']:+.2f} $)"
        )
    if worst_trade is not None:
        st.write(
            f"**Worst trade:** {worst_trade['ticker']} "
            f"({worst_trade['pnl_dollars']:+.2f} $)"
        )

    # Daily P&L bar chart
    if not taken.empty:
        daily_pnl = (
            taken.assign(day=taken["timestamp"].dt.date)
            .groupby("day")["pnl_dollars"]
            .sum()
            .reset_index()
        )
        daily_pnl["color"] = np.where(daily_pnl["pnl_dollars"] >= 0, "#16a34a", "#dc2626")
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=daily_pnl["day"],
                    y=daily_pnl["pnl_dollars"],
                    marker_color=daily_pnl["color"],
                )
            ]
        )
        fig_bar.update_layout(
            template="plotly_dark",
            title="Daily P&L (this week)",
            xaxis_title="Day",
            yaxis_title="P&L ($)",
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Equity curve with optional starting balance
        start_balance = st.number_input(
            "Starting balance for equity curve ($)",
            min_value=0.0,
            value=10000.0,
            step=100.0,
        )
        equity = start_balance + daily_pnl["pnl_dollars"].cumsum()
        fig_eq = go.Figure(
            data=[
                go.Scatter(
                    x=daily_pnl["day"],
                    y=equity,
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=2),
                )
            ]
        )
        fig_eq.update_layout(
            template="plotly_dark",
            title="Running account balance (this week)",
            xaxis_title="Day",
            yaxis_title="Balance ($)",
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

# === Dobby AI analyst chat ===
st.divider()
st.subheader("Ask Dobby")

if not st.session_state["dobby_chat"]:
    st.session_state["dobby_chat"].append(
        {
            "role": "assistant",
            "content": (
                "Hi Master, I'm Dobby your personal trading analyst. "
                "Ask me anything about today's opportunities, market conditions, or your positions."
            ),
        }
    )

for msg in st.session_state["dobby_chat"][-5:]:
    if msg["role"] == "assistant":
        with st.chat_message("Dobby ⭐"):
            st.write(msg["content"])
    else:
        with st.chat_message("Master"):
            st.write(msg["content"])

prompt = st.chat_input("Ask Dobby anything about today's trades...")


def _dobby_reply(text: str) -> str:
    now_uk = _now_uk()
    in_session = _in_market_hours_uk(now_uk)
    results = None
    if "results" in globals():
        results = globals()["results"]

    # Out of hours behaviour
    if not in_session:
        return (
            "Markets are currently closed Master. I can still help you review your journal, "
            "analyse past trades, or prepare for tomorrow's session."
        )

    # Simple intent heuristics
    lower = text.lower()
    pieces = []

    if "best" in lower and "trade" in lower:
        if results:
            opps = results.get("opportunities", []) or []
            if opps:
                best = opps[0]
                pieces.append(
                    f"Good morning Master, the strongest opportunity right now is {best['ticker']} "
                    f"with a {best['action']} bias and risk/reward of about {best['risk_reward']:.1f}:1."
                )
            else:
                pieces.append(
                    "Master, I don't see a clear standout trade at the moment — signals are mixed."
                )
        else:
            pieces.append("Master, I have no fresh scan results yet. Run a scan and I'll review them.")
    elif "why" in lower and "moving" in lower:
        ticker = None
        for t in TICKER_BAR:
            if t.lower() in lower:
                ticker = t
                break
        if results and ticker:
            opps = results.get("opportunities", []) or []
            match = next((o for o in opps if o["ticker"] == ticker), None)
            if match:
                fr = ", ".join(match.get("frameworks", [])) or "no major frameworks"
                pieces.append(
                    f"Master, {ticker} is moving because multiple signals align — "
                    f"timeframes {', '.join(match.get('timeframes_confirmed', []))} and frameworks {fr}."
                )
            else:
                pieces.append(
                    f"Master, {ticker} is active but not one of today's top opportunities in my scan."
                )
        else:
            pieces.append(
                "Master, I can't tie that move to today's scan yet — data may be stale or missing."
            )
    else:
        if results:
            opps = results.get("opportunities", []) or []
            macro = results.get("macro", {}) or {}
            note = macro.get("note", "")
            if opps:
                pieces.append(
                    f"Master, today's scan found {len(opps)} opportunities. "
                    f"The strongest ideas lean {opps[0]['action']} with solid multi-timeframe confirmation."
                )
            if note:
                pieces.append(f"Macro context: {note}")
        else:
            pieces.append(
                "Master, I don't have fresh opportunities loaded yet. Run a scan and I'll summarise them for you."
            )

    return " ".join(pieces)


if prompt:
    st.session_state["dobby_chat"].append({"role": "user", "content": prompt})
    reply = _dobby_reply(prompt)
    st.session_state["dobby_chat"].append({"role": "assistant", "content": reply})

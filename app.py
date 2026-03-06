"""
app.py  –  The Trading Floor
═════════════════════════════
Gambling-themed AI paper trading simulator.

Run:
    streamlit run app.py
"""

import logging
import random
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents import DataAgent, StrategyAgent, RiskAgent, ExecutionAgent

logger = logging.getLogger(__name__)
from config import (
    API_KEY,
    CSV_PATH,
    INITIAL_CAPITAL,
    LOOKBACK_DAYS,
    MAX_POSITION_PCT,
    MAX_TRADES_PER_DAY,
    SIMULATION_DAYS,
)
from database import PortfolioDatabase
import market_data as md


# ═══════════════════════════════════════════════════════════════════════════════
# Page config & theme
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="The Trading Floor",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Metric labels – gold uppercase ── */
[data-testid="stMetricLabel"] p {
    color: #f59e0b !important;
    font-size: 0.68rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Courier New', monospace !important;
    font-size: 1.7rem !important;
    font-weight: 900 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d14 0%, #0c1520 100%) !important;
    border-right: 1px solid rgba(245,158,11,0.25) !important;
}
section[data-testid="stSidebar"] .stButton button {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] p {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-size: 0.78rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] p { color: #f59e0b !important; }

/* ── Dividers ── */
hr { border-color: rgba(245,158,11,0.18) !important; }

/* ── Captions / secondary text ── */
.stCaption { color: #6b7280 !important; font-size: 0.72rem !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid rgba(245,158,11,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ordinal(n: int) -> str:
    """Return n with its English ordinal suffix, e.g. 1 → '1st', 5 → '5th'."""
    if 11 <= (n % 100) <= 13:          # special cases: 11th, 12th, 13th
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th','th','th','th','th','th'][n % 10]}"


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading price data …")
def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
# Single-day simulation
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_day(
    prices_df: pd.DataFrame,
    sim_date:  pd.Timestamp,
    strategy:  str,
    db:        PortfolioDatabase,
) -> dict[str, Any]:
    """Run one day through all four agents and return the day's result dict."""
    date_str  = sim_date.strftime("%Y-%m-%d")
    prices    = prices_df.loc[sim_date].to_dict()
    portfolio = db.get_portfolio_state(prices)

    data_agent      = DataAgent()
    strategy_agent  = StrategyAgent(strategy)
    risk_agent      = RiskAgent()
    execution_agent = ExecutionAgent(db)

    analysis    = data_agent.analyze(prices_df, sim_date)
    proposed    = strategy_agent.propose_trades(analysis, portfolio, date_str,
                                                 st.session_state.get("sim_day_num", 1))
    approved    = risk_agent.validate(proposed, portfolio, prices)
    exec_report = execution_agent.execute(approved, prices, date_str)

    return {
        "date":            date_str,
        "day_num":         st.session_state.get("sim_day_num", 1),
        "portfolio":       exec_report["portfolio"],
        "analysis":        analysis,
        "proposed_trades": proposed,
        "approved_trades": exec_report["executed_trades"],
        "violations":      risk_agent.violations.copy(),
        "llm_reasoning":   strategy_agent.reasoning,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════════

_DARK = "plotly_dark"


def chart_equity_curve(history: pd.DataFrame, initial_capital: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["date"], y=history["total_equity"],
        name="Portfolio Value",
        mode="lines+markers",
        line=dict(color="#f59e0b", width=3),
        marker=dict(size=8, color="#f59e0b"),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.06)",
    ))
    fig.add_hline(
        y=initial_capital,
        line_dash="dot", line_color="#4b5563",
        annotation_text=f"Start  ${initial_capital:,.0f}",
        annotation_position="bottom right",
        annotation_font_color="#6b7280",
    )
    fig.update_layout(
        title=dict(text="BANKROLL OVER TIME", font=dict(color="#f59e0b", size=12)),
        xaxis_title=None, yaxis_title="Value (USD)",
        template=_DARK, hovermode="x unified",
        height=360, margin=dict(t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def chart_allocation(portfolio: dict) -> go.Figure:
    import plotly.express as px
    labels = ["Cash"] + list(portfolio["positions"].keys())
    values = [portfolio["cash"]] + [p["value"] for p in portfolio["positions"].values()]
    colors = ["#374151"] + ["#f59e0b", "#10b981", "#3b82f6", "#8b5cf6",
                             "#ef4444", "#06b6d4", "#f97316", "#84cc16",
                             "#e879f9", "#fb7185"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker_colors=colors[:len(labels)],
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig.update_layout(
        title=dict(text="ALLOCATION", font=dict(color="#f59e0b", size=12)),
        template=_DARK, height=320,
        showlegend=False,
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Tab renders
# ═══════════════════════════════════════════════════════════════════════════════

def render_trading_floor(daily_results: list[dict], strategy_lbl: str, portfolio_report: dict | None):
    # ── Day 0: no simulation started — show welcome card only ────────────────
    if not daily_results:
        st.markdown(
            f"""
            <div style='text-align:center; padding: 4rem 2rem;'>
                <div style='font-size:4rem; margin-bottom:1rem;'>🎰</div>
                <div style='font-size:0.72rem; font-weight:800; text-transform:uppercase;
                            letter-spacing:0.16em; color:#f59e0b; margin-bottom:0.5rem;'>
                    Welcome to The Trading Floor
                </div>
                <div style='font-size:2.4rem; font-weight:900; font-family:monospace;
                            color:#f1f5f9; margin-bottom:0.4rem;'>
                    ${INITIAL_CAPITAL:,.0f}
                </div>
                <div style='font-size:0.95rem; color:#6b7280; max-width:420px;
                            margin:0 auto; line-height:1.6;'>
                    Your starting bankroll is ready.<br>
                    Hit <strong style='color:#f59e0b;'>START NEW GAME</strong>
                    in the sidebar, then
                    <strong style='color:#f59e0b;'>DEAL NEXT HAND</strong>
                    to let the AI play the market.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Daily Briefing (shown after at least one day has run) ────────────────
    if portfolio_report:
        r = portfolio_report
        with st.container():
            st.markdown(
                f"<div style='font-size:0.65rem; font-weight:800; text-transform:uppercase; "
                f"letter-spacing:0.14em; color:#f59e0b;'>📋 Daily Briefing  ·  {r['generated_date']}</div>",
                unsafe_allow_html=True,
            )

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Portfolio Value",  f"${r['portfolio_value']:,.0f}")
            b2.metric("All-Time P/L",
                      f"{r['alltime_pnl_pct']:+.2f}%",
                      delta=f"${r['alltime_pnl']:+,.0f}")
            b3.metric("5-Day P/L",
                      f"{r['fiveday_pnl_pct']:+.2f}%",
                      delta=f"${r['fiveday_pnl']:+,.0f}")
            b4.metric("Win Rate",  f"{r['win_rate']:.0f}%",
                      help="% of trading days portfolio increased")

            with st.expander("📰 AI Market Summary", expanded=True):
                st.markdown(
                    f"<div style='line-height:1.7; color:#d1d5db; font-size:0.9rem;'>"
                    f"{r['summary'].replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True,
                )

                # News headlines
                if r.get("news_headlines"):
                    st.divider()
                    st.markdown(
                        "<div style='font-size:0.65rem; font-weight:800; text-transform:uppercase; "
                        "letter-spacing:0.12em; color:#f59e0b;'>Recent Headlines</div>",
                        unsafe_allow_html=True,
                    )
                    for ticker, headlines in r["news_headlines"].items():
                        for h in headlines:
                            st.markdown(
                                f"<div style='font-size:0.8rem; color:#9ca3af; "
                                f"padding:2px 0;'>▸ <strong style='color:#d1d5db;'>"
                                f"{ticker}</strong> — {h}</div>",
                                unsafe_allow_html=True,
                            )

        st.divider()

    day = daily_results[-1]

    st.markdown(
        f"<div style='font-size:0.72rem; font-weight:800; text-transform:uppercase; "
        f"letter-spacing:0.12em; color:#f59e0b; margin-bottom:0.5rem;'>"
        f"Day {day['day_num']}  ·  {day['date']}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])

    # ── DataAgent output ──────────────────────────────────────────────────────
    with left:
        st.markdown("**Intelligence Report** *(Data Agent)*")
        rows = []
        for t, m in day["analysis"].items():
            rows.append({
                "Ticker":   t,
                "Price":    f"${m['price']:.2f}",
                "Mom 20d":  f"{m['momentum_20d']:+.2f}%",
                "Mom 5d":   f"{m['momentum_5d']:+.2f}%",
                "Z-Score":  f"{m['zscore']:+.3f}",
                "RSI":      f"{m['rsi']:.1f}",
                "vs SMA20": "▲" if m["above_sma20"] else "▼",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=280)

    # ── Strategy + Risk + Execution output ───────────────────────────────────
    with right:
        st.markdown("**Strategy Agent's Play**")

        approved_keys = {(t["ticker"], t["action"]) for t in day["approved_trades"]}

        if not day["proposed_trades"]:
            st.info("No trades proposed today.")
        else:
            for prop in day["proposed_trades"]:
                key     = (prop["ticker"], prop["action"])
                is_buy  = prop["action"] == "BUY"
                icon    = "🟢" if is_buy else "🔴"
                verdict = "✅ executed" if key in approved_keys else "❌ blocked"
                color   = "#10b981" if is_buy else "#ef4444"
                st.markdown(
                    f"{icon} <span style='color:{color}; font-weight:700;'>"
                    f"{prop['action']} {prop['ticker']}</span> &nbsp; {verdict}",
                    unsafe_allow_html=True,
                )
                st.caption(prop.get("reasoning", ""))

        if day["violations"]:
            st.markdown("**Risk Agent — Blocked Plays**")
            for v in day["violations"]:
                st.warning(v, icon="⚠️")

        if day.get("llm_reasoning"):
            with st.expander("Raw LLM output"):
                st.code(day["llm_reasoning"], language="json")

    st.divider()

    # ── History of all days ───────────────────────────────────────────────────
    if len(daily_results) > 1:
        st.markdown("**Previous Rounds**")
        for past in reversed(daily_results[:-1]):
            n_exec = len(past["approved_trades"])
            badge  = f"{n_exec} trade(s)" if n_exec else "no trades"
            with st.expander(f"Day {past['day_num']}  ·  {past['date']}  ·  {badge}"):
                if past["approved_trades"]:
                    rows = []
                    for t in past["approved_trades"]:
                        rows.append({
                            "Action": t["action"],
                            "Ticker": t["ticker"],
                            "Qty":    f"{t['quantity']:.4f}",
                            "Price":  f"${t['price']:.2f}",
                            "Amount": f"${t['amount']:,.2f}",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.caption("No trades executed.")


def render_portfolio(db: PortfolioDatabase, daily_results: list[dict]):
    if not daily_results:
        st.info("Start a simulation to see your portfolio.")
        return

    latest  = daily_results[-1]
    snap    = latest["portfolio"]
    prices  = {t: m["price"] for t, m in latest["analysis"].items()}

    # ── Holdings ─────────────────────────────────────────────────────────────
    st.markdown("**Holdings**")
    if snap["positions"]:
        rows = []
        for t, p in snap["positions"].items():
            pnl_val = p["value"] - p["shares"] * p["average_cost"]
            rows.append({
                "Ticker":    t,
                "Qty":       round(p["shares"], 4),
                "Avg Cost":  f"${p['average_cost']:.2f}",
                "Last Px":   f"${p['current_price']:.2f}",
                "Mkt Value": f"${p['value']:,.2f}",
                "P/L $":     f"${pnl_val:+,.2f}",
                "P/L %":     f"{p['pnl_pct']:+.2f}%",
            })
        df_pos = pd.DataFrame(rows)

        def _color_pnl(val):
            if not isinstance(val, str):
                return ""
            if val.startswith("+"):
                return "color:#10b981; font-weight:700"
            if val.startswith("-"):
                return "color:#ef4444; font-weight:700"
            return ""

        st.dataframe(
            df_pos.style.map(_color_pnl, subset=["P/L $", "P/L %"]),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info(f"No open positions.  Bankroll: ${snap['cash']:,.2f}")

    st.divider()

    col_risk, col_chart = st.columns([1, 1])

    # ── Risk summary ─────────────────────────────────────────────────────────
    with col_risk:
        st.markdown("**Position Limits / Risk Summary**")
        trades_today = len(daily_results[-1]["approved_trades"])
        total_val    = snap["total_value"]

        risk_rows = [
            {"Rule":    "Max single position",
             "Limit":   f"{MAX_POSITION_PCT*100:.0f}%",
             "Status":  "✅ OK"},
            {"Rule":    "Trades today",
             "Limit":   f"{MAX_TRADES_PER_DAY}",
             "Status":  f"{trades_today} / {MAX_TRADES_PER_DAY}  "
                        f"{'✅' if trades_today <= MAX_TRADES_PER_DAY else '❌'}"},
            {"Rule":    "Cash remaining",
             "Limit":   "—",
             "Status":  f"${snap['cash']:,.0f}  ({snap['cash']/total_val*100:.1f}%)"},
        ]
        # Per-position concentration check
        for t, p in snap["positions"].items():
            wt = p["value"] / total_val * 100
            risk_rows.append({
                "Rule":   f"{t} weight",
                "Limit":  f"{MAX_POSITION_PCT*100:.0f}%",
                "Status": f"{wt:.1f}%  {'✅' if wt <= MAX_POSITION_PCT*100 else '🚨 OVER'}",
            })

        st.dataframe(pd.DataFrame(risk_rows), hide_index=True, use_container_width=True)

    # ── Allocation pie ────────────────────────────────────────────────────────
    with col_chart:
        st.plotly_chart(chart_allocation(snap), use_container_width=True)

    st.divider()

    # ── Position History (open + closed) ──────────────────────────────────────
    st.markdown("**Position History** *(open & closed positions with dates)*")
    hist_df = db.get_position_history()

    if hist_df.empty:
        st.info("No position history yet — start a simulation and execute some trades.")
    else:
        phdisp = hist_df.copy()

        # Status badge
        phdisp["Status"] = phdisp["closed_date"].apply(
            lambda x: "🟢 Open" if (pd.isna(x) or x == "" or x is None) else "🔴 Closed"
        )

        # Format numeric columns safely
        def _fmt_price(v):
            try:
                return f"${float(v):.2f}" if pd.notna(v) else "—"
            except (TypeError, ValueError):
                return "—"

        def _fmt_pnl(v):
            try:
                f = float(v)
                return f"${f:+,.2f}" if pd.notna(v) else "—"
            except (TypeError, ValueError):
                return "—"

        def _fmt_pct(v):
            try:
                f = float(v)
                return f"{f:+.2f}%" if pd.notna(v) else "—"
            except (TypeError, ValueError):
                return "—"

        phdisp["Avg Cost"]    = phdisp["avg_cost"].map(_fmt_price)
        phdisp["Exit Price"]  = phdisp["close_price"].map(_fmt_price)
        phdisp["Realized P/L"]= phdisp["realized_pnl"].map(_fmt_pnl)
        phdisp["P/L %"]       = phdisp["realized_pnl_pct"].map(_fmt_pct)
        phdisp["Closed"]      = phdisp["closed_date"].apply(
            lambda x: x if (pd.notna(x) and x not in ("", None)) else "—"
        )

        ph_show = phdisp[[
            "symbol", "opened_date", "Closed", "shares",
            "Avg Cost", "Exit Price", "Realized P/L", "P/L %", "Status",
        ]].copy()
        ph_show.columns = [
            "Ticker", "Opened", "Closed", "Shares",
            "Avg Cost", "Exit Price", "Realized P/L", "P/L %", "Status",
        ]

        def _color_pnl_str(val):
            if not isinstance(val, str):
                return ""
            if val.startswith("+") or (val.startswith("$+") if val.startswith("$") else False):
                return "color:#10b981; font-weight:700"
            if val.startswith("-") or (val.startswith("$-") if val.startswith("$") else False):
                return "color:#ef4444; font-weight:700"
            # handle "$+..." and "$-..."
            if len(val) > 1 and val[1] in ("+", "-"):
                return "color:#10b981; font-weight:700" if val[1] == "+" else "color:#ef4444; font-weight:700"
            return ""

        st.dataframe(
            ph_show.style.map(_color_pnl_str, subset=["Realized P/L", "P/L %"]),
            hide_index=True, use_container_width=True,
        )

    st.divider()

    # ── Trade Log ─────────────────────────────────────────────────────────────
    st.markdown("**Trade Log** *(all executions)*")
    trades_df = db.get_trades()

    if trades_df.empty:
        st.info("No trades executed yet.")
    else:
        def _color_action(val):
            if val == "BUY":
                return "color:#10b981; font-weight:700"
            return "color:#ef4444; font-weight:700"

        # Include `amount` column if it exists (added in updated schema)
        cols_wanted = ["date", "action", "symbol", "shares", "price", "amount", "reason"]
        cols_avail  = [c for c in cols_wanted if c in trades_df.columns]
        display = trades_df[cols_avail].copy()

        col_labels = {
            "date":   "Date",
            "action": "Action",
            "symbol": "Ticker",
            "shares": "Qty",
            "price":  "Price",
            "amount": "Total",
            "reason": "Reasoning",
        }
        display.rename(columns=col_labels, inplace=True)

        if "Price" in display.columns:
            display["Price"] = display["Price"].map("${:.2f}".format)
        if "Total" in display.columns:
            display["Total"] = display["Total"].apply(
                lambda v: f"${float(v):,.2f}" if pd.notna(v) else "—"
            )

        st.dataframe(
            display.style.map(_color_action, subset=["Action"]),
            hide_index=True, use_container_width=True, height=300,
        )


def render_performance(db: PortfolioDatabase, initial_capital: float, daily_results: list[dict]):
    history_df = db.get_portfolio_history()

    if history_df.empty:
        st.info("No performance data yet — run at least one day.")
        return

    # ── KPI calculations ──────────────────────────────────────────────────────
    equity    = history_df["total_equity"]
    final_val = equity.iloc[-1]
    pnl_pct   = (final_val / initial_capital - 1.0) * 100.0
    peak      = equity.max()
    trough    = equity.min()
    max_dd    = (peak - trough) / peak * 100.0 if peak > 0 else 0.0

    # Win rate: % of days portfolio increased
    daily_changes = equity.diff().dropna()
    win_rate = (daily_changes > 0).sum() / len(daily_changes) * 100.0 if len(daily_changes) > 0 else 0.0

    num_trades = len(db.get_trades())

    # ── KPI cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Return",    f"{pnl_pct:+.2f}%",
              delta=f"${final_val - initial_capital:+,.0f}")
    k2.metric("Final Bankroll",  f"${final_val:,.2f}")
    k3.metric("Peak Value",      f"${peak:,.2f}")
    k4.metric("Max Drawdown",    f"{max_dd:.2f}%")
    k5.metric("Win Rate",        f"{win_rate:.0f}%",
              help="% of trading days where portfolio increased")

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.plotly_chart(chart_equity_curve(history_df, initial_capital), use_container_width=True)

    st.divider()

    # ── Daily snapshots table ─────────────────────────────────────────────────
    st.markdown("**Daily Snapshots**")

    rows = []
    for i, day in enumerate(daily_results):
        snap = day["portfolio"]
        prev_eq = daily_results[i-1]["portfolio"]["total_value"] if i > 0 else initial_capital
        day_pnl = snap["total_value"] - prev_eq
        rows.append({
            "Date":      day["date"],
            "Cash":      f"${snap['cash']:,.2f}",
            "Holdings":  f"${snap['positions_value']:,.2f}",
            "Total":     f"${snap['total_value']:,.2f}",
            "Day P/L":   f"${day_pnl:+,.2f}",
            "Trades":    len(day["approved_trades"]),
            "Notes":     ", ".join(
                             f"{t['action']} {t['ticker']}"
                             for t in day["approved_trades"]
                         ) or "—",
        })

    if not rows:
        st.info("Run at least one day in the current session to see the snapshot table.")
        return

    df_snap = pd.DataFrame(rows)

    def _color_pnl(val):
        if not isinstance(val, str):
            return ""
        if val.startswith("+"):
            return "color:#10b981; font-weight:700"
        if val.startswith("-"):
            return "color:#ef4444; font-weight:700"
        return ""

    st.dataframe(
        df_snap.style.map(_color_pnl, subset=["Day P/L"]),
        hide_index=True, use_container_width=True,
    )


def render_market_data(db: PortfolioDatabase):
    st.markdown(
        "<div style='font-size:0.72rem; font-weight:800; text-transform:uppercase; "
        "letter-spacing:0.12em; color:#f59e0b; margin-bottom:0.75rem;'>"
        "S&P 500 Live Market Data</div>",
        unsafe_allow_html=True,
    )
    st.caption("Fetches all S&P 500 constituents with latest OHLCV via yfinance. Cached in local DB.")

    fetch_dates = db.get_sp500_fetch_dates()
    col_btn, col_date = st.columns([1, 3])

    with col_btn:
        fetch_btn = st.button("Fetch / Refresh", type="primary", use_container_width=True)

    if fetch_btn:
        with st.spinner("Pulling quotes from Yahoo Finance …"):
            try:
                md.fetch_and_store_sp500(db)
                st.success("Market data refreshed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed: {exc}")
                return

    fetch_dates = db.get_sp500_fetch_dates()
    if not fetch_dates:
        st.info("No data yet — press **Fetch / Refresh** to download today's quotes.")
        return

    with col_date:
        selected_date = st.selectbox(
            "Snapshot", options=fetch_dates, index=0, label_visibility="collapsed"
        )

    df = db.get_sp500_stocks(selected_date)
    if df.empty:
        st.warning("No data for this date.")
        return

    df["change_pct"] = ((df["close"] - df["open"]) / df["open"] * 100).round(2)

    total   = len(df)
    gainers = int((df["change_pct"] > 0).sum())
    losers  = int((df["change_pct"] < 0).sum())

    # Derive the actual market date of the price data (may lag fetch_date on weekends)
    price_date = selected_date  # fallback
    if "price_date" in df.columns:
        _pd_vals = df["price_date"].dropna()
        if not _pd_vals.empty:
            price_date = str(_pd_vals.iloc[0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks",       str(total))
    m2.metric("Winners",      str(gainers), delta=f"+{gainers}")
    m3.metric("Losers",       str(losers),  delta=f"-{losers}", delta_color="inverse")
    m4.metric("Market Date",  price_date,
              help=f"Actual trading date of the OHLCV data · Fetched: {selected_date}")

    if price_date != selected_date:
        st.caption(
            f"ℹ️  Prices quoted as of **{price_date}** (last trading day) — "
            f"fetched on {selected_date}."
        )

    st.divider()

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("Search ticker or company", placeholder="e.g. AAPL or Apple")
    with fc2:
        sectors = ["All Sectors"] + sorted(df["sector"].dropna().unique().tolist())
        sector_filter = st.selectbox("Sector", sectors)
    with fc3:
        direction = st.selectbox("Direction", ["All", "Winners only", "Losers only"])

    filtered = df.copy()
    if search:
        q = search.upper()
        filtered = filtered[
            filtered["symbol"].str.upper().str.contains(q, na=False)
            | filtered["name"].str.upper().str.contains(q, na=False)
        ]
    if sector_filter != "All Sectors":
        filtered = filtered[filtered["sector"] == sector_filter]
    if direction == "Winners only":
        filtered = filtered[filtered["change_pct"] > 0]
    elif direction == "Losers only":
        filtered = filtered[filtered["change_pct"] < 0]

    st.caption(f"Showing {len(filtered)} of {total} stocks")

    display = filtered[["symbol", "name", "sector", "open", "high", "low",
                         "close", "change_pct", "volume"]].copy()
    display.columns = ["Symbol", "Company", "Sector", "Open", "High", "Low",
                        "Close", "Change %", "Volume"]

    def _style_change(val):
        if pd.isna(val):
            return ""
        return "color:#10b981; font-weight:700" if val > 0 else "color:#ef4444; font-weight:700"

    st.dataframe(
        display.style
        .map(_style_change, subset=["Change %"])
        .format({
            "Open":     lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "High":     lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "Low":      lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "Close":    lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "Change %": lambda v: f"{v:+.2f}%" if pd.notna(v) else "—",
            "Volume":   lambda v: f"{int(v):,}" if pd.notna(v) else "—",
        }),
        hide_index=True, use_container_width=True, height=600,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Roulette mini-game  (sidebar, just for fun — no portfolio effect)
# ═══════════════════════════════════════════════════════════════════════════════

_RED_NUMBERS   = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
_BLACK_NUMBERS = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}

def _spin_roulette(pick: str):
    """Spin the wheel, update session state, return (number, color, won)."""
    number = random.randint(0, 36)
    if number == 0:
        color = "green"
    elif number in _RED_NUMBERS:
        color = "red"
    else:
        color = "black"
    won = (color == pick)

    if won:
        st.session_state["roulette_wins"] += 1
        st.session_state["roulette_streak"] = max(st.session_state["roulette_streak"], 0) + 1
    else:
        st.session_state["roulette_losses"] += 1
        st.session_state["roulette_streak"] = min(st.session_state["roulette_streak"], 0) - 1

    st.session_state["roulette_result"] = {"number": number, "color": color, "won": won, "pick": pick}


def render_roulette():
    """Renders the roulette mini-game inside the sidebar."""
    st.markdown(
        "<div style='font-size:0.65rem; font-weight:800; text-transform:uppercase; "
        "letter-spacing:0.14em; color:#f59e0b; margin-bottom:0.6rem;'>🎡 Roulette</div>",
        unsafe_allow_html=True,
    )

    # ── Bet selection ─────────────────────────────────────────────────────────
    pick_label = st.radio(
        "bet",
        ["🔴  Red", "⚫  Black", "🟢  Green"],
        horizontal=True,
        label_visibility="collapsed",
        key="roulette_pick_radio",
    )
    pick = pick_label.split()[1].lower()     # "red" / "black" / "green"

    # Payout reminder
    payouts = {"red": "2×", "black": "2×", "green": "35×"}
    st.markdown(
        f"<div style='font-size:0.68rem; color:#6b7280; text-align:center; "
        f"margin-bottom:0.5rem;'>Payout: <strong style='color:#f59e0b;'>{payouts[pick]}</strong> "
        f"&nbsp;·&nbsp; Green odds: 1 in 37</div>",
        unsafe_allow_html=True,
    )

    # ── Spin button ───────────────────────────────────────────────────────────
    if st.button("🎰  Spin the Wheel", type="primary", use_container_width=True):
        with st.spinner("🎡  Spinning…"):
            import time; time.sleep(0.6)
        _spin_roulette(pick)
        st.rerun()

    # ── Result display ────────────────────────────────────────────────────────
    result = st.session_state.get("roulette_result")
    if result:
        num    = result["number"]
        color  = result["color"]
        won    = result["won"]
        r_pick = result["pick"]

        bg = {"red": "#991b1b", "black": "#111827", "green": "#14532d"}[color]
        border = "#f59e0b" if won else "#374151"
        outcome_color = "#10b981" if won else "#ef4444"
        outcome_text  = "YOU WIN! 🎉" if won else "YOU LOSE 💸"

        st.markdown(
            f"""
            <div style='text-align:center; margin:0.75rem 0; padding:0.9rem 0.5rem;
                        background:{bg}22; border:1px solid {border};
                        border-radius:10px;'>
                <div style='font-size:2.2rem; font-weight:900; font-family:monospace;
                             color:white; line-height:1;
                             background:{bg}; display:inline-flex;
                             align-items:center; justify-content:center;
                             width:62px; height:62px; border-radius:50%;
                             border:3px solid {border};'>
                    {num}
                </div>
                <div style='font-size:0.72rem; color:#9ca3af; margin-top:0.4rem;
                             text-transform:uppercase; letter-spacing:0.1em;'>
                    {color.upper()}
                    &nbsp;·&nbsp; you bet {r_pick.upper()}
                </div>
                <div style='font-size:1rem; font-weight:900; color:{outcome_color};
                             margin-top:0.35rem; text-transform:uppercase;
                             letter-spacing:0.08em;'>
                    {outcome_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Win / loss counter ────────────────────────────────────────────────────
    wins    = st.session_state["roulette_wins"]
    losses  = st.session_state["roulette_losses"]
    streak  = st.session_state["roulette_streak"]

    if wins + losses > 0:
        streak_label = (
            f"🔥 {streak}W"  if streak > 0 else
            f"❄️ {abs(streak)}L" if streak < 0 else "—"
        )
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; "
            f"font-size:0.72rem; color:#6b7280; padding:0 0.2rem;'>"
            f"<span>🟢 W: <strong style='color:#10b981;'>{wins}</strong></span>"
            f"<span>🔴 L: <strong style='color:#ef4444;'>{losses}</strong></span>"
            f"<span>Streak: <strong style='color:#f59e0b;'>{streak_label}</strong></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("Reset Stats", use_container_width=True):
            st.session_state["roulette_wins"]   = 0
            st.session_state["roulette_losses"] = 0
            st.session_state["roulette_streak"] = 0
            st.session_state["roulette_result"] = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    db = PortfolioDatabase()

    # Load price data only if the CSV exists (created by Start New Game)
    _csv_ready = Path(CSV_PATH).exists()
    prices_df  = load_price_data() if _csv_ready else None

    # ── Session state init ────────────────────────────────────────────────────
    defaults = {
        "sim_started":       False,
        "sim_date_idx":      0,
        "sim_day_num":       1,
        "sim_dates":         [],
        "daily_results":     [],
        "sim_capital":       INITIAL_CAPITAL,
        "sim_strategy":      "Momentum",
        "portfolio_report":  None,
        "report_date":       "",
        # roulette
        "roulette_wins":     0,
        "roulette_losses":   0,
        "roulette_streak":   0,
        "roulette_result":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Auto-generate portfolio report once per calendar day ─────────────────
    # Only runs after Start New Game has populated the price CSV
    today_str = date.today().isoformat()
    if _csv_ready and st.session_state["report_date"] != today_str:
        with st.spinner("Generating daily briefing …"):
            try:
                data_agent = DataAgent()
                st.session_state["portfolio_report"] = data_agent.generate_portfolio_report(
                    db, prices_df
                )
                st.session_state["report_date"] = today_str
            except Exception as exc:
                st.session_state["portfolio_report"] = None
                logger.warning("Portfolio report failed: %s", exc)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='font-size:1.3rem; font-weight:900; color:#f59e0b; "
            "text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.25rem;'>"
            "🎰 The Trading Floor</div>",
            unsafe_allow_html=True,
        )
        st.caption("AI-powered paper trading simulator")
        st.divider()

        # Current date display — "5th March 2026  |  Day: 1"
        _today     = date.today()
        _date_part = f"{_ordinal(_today.day)} {_today.strftime('%B %Y')}"

        if st.session_state["sim_started"] and st.session_state["daily_results"]:
            day_num = st.session_state["sim_day_num"] - 1
            st.markdown(
                f"<div style='font-size:1.05rem; font-weight:900; font-family:monospace; "
                f"color:#f59e0b; line-height:1.3;'>"
                f"{_date_part}"
                f"<span style='color:#6b7280; font-weight:400;'>&nbsp; | &nbsp;</span>"
                f"Day: {day_num}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:1.05rem; font-weight:900; font-family:monospace; "
                f"color:#f59e0b; line-height:1.3;'>"
                f"{_date_part}"
                f"<span style='color:#6b7280; font-weight:400;'>&nbsp; | &nbsp;</span>"
                f"Day: —"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Roulette mini-game ─────────────────────────────────────────────
        render_roulette()

        st.divider()

        # ── Simulation controls ────────────────────────────────────────────
        # ── Start New Game button ──────────────────────────────────────────
        if st.button("🎲  Start New Game", type="primary", use_container_width=True):
            with st.spinner("Wiping data and fetching fresh S&P 500 data from Yahoo Finance …"):
                try:
                    # 1. Full DB wipe (all tables)
                    db.reset_all(INITIAL_CAPITAL)
                    # 2. Pull real S&P 500 price history and overwrite the CSV
                    fresh_df = md.fetch_sp500_prices_csv(CSV_PATH)
                    # 3. Fetch today's OHLCV snapshot into the sp500_stocks table
                    md.fetch_and_store_sp500(db)
                    # 4. Bust the cache so the new CSV is picked up on rerun
                    st.cache_data.clear()
                except Exception as exc:
                    st.error(f"Failed to fetch market data: {exc}")
                    st.stop()

            # Pick the last SIMULATION_DAYS trading days from the fresh data
            fresh_df.index = pd.to_datetime(fresh_df.index)
            fresh_df = fresh_df.sort_index()
            sim_dates = fresh_df.index[-SIMULATION_DAYS:].tolist()

            st.session_state.update({
                "sim_started":      True,
                "sim_date_idx":     0,
                "sim_day_num":      1,
                "sim_dates":        sim_dates,
                "daily_results":    [],
                "sim_capital":      INITIAL_CAPITAL,
                "sim_strategy":     "Momentum",
                "report_date":      "",
                "portfolio_report": None,
            })
            st.rerun()

        # ── Deal Next Hand button ──────────────────────────────────────────
        already_ran_today = db.get_last_advance_date() == today_str
        dates_remaining   = (
            st.session_state["sim_started"]
            and st.session_state["sim_date_idx"] < len(st.session_state["sim_dates"])
        )
        # prices_df is always available when sim_started (Start New Game fetches it first)
        can_advance = dates_remaining and not already_ran_today and prices_df is not None

        if already_ran_today and dates_remaining:
            st.markdown(
                "<div style='font-size:0.72rem; color:#f59e0b; font-weight:700; "
                "text-align:center; padding:0.5rem 0;'>"
                "⏳ Already ran today — come back tomorrow</div>",
                unsafe_allow_html=True,
            )

        if st.button(
            "📈  Deal Next Hand",
            disabled=not can_advance,
            use_container_width=True,
        ):
            idx          = st.session_state["sim_date_idx"]
            sim_date     = st.session_state["sim_dates"][idx]
            strategy_now = st.session_state["sim_strategy"]

            with st.spinner(f"Running Day {st.session_state['sim_day_num']} …"):
                result = run_single_day(prices_df, sim_date, strategy_now, db)

            db.set_last_advance_date(today_str)
            st.session_state["daily_results"].append(result)
            st.session_state["sim_date_idx"] += 1
            st.session_state["sim_day_num"]  += 1
            # Regenerate report after advancing
            st.session_state["report_date"] = ""
            st.rerun()

        if st.session_state["sim_started"] and not dates_remaining:
            st.success("Simulation complete — all hands dealt.", icon="🏁")

        st.divider()
        if API_KEY:
            st.success("Claude API: connected", icon="🤖")
        else:
            st.warning("No API key — rule-based mode", icon="⚙️")

    # ── Top KPI bar ───────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-size:1.6rem; font-weight:900; color:#f1f5f9; "
        "text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0;'>"
        "AI Investment Application"
        "<span style='color:#f59e0b;'> · Paper Trading Simulator</span></h1>",
        unsafe_allow_html=True,
    )

    # Live KPI bar
    if st.session_state["daily_results"]:
        snap     = st.session_state["daily_results"][-1]["portfolio"]
        init_cap = st.session_state["sim_capital"]
        pnl      = snap["total_value"] - init_cap
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Portfolio Value", f"${snap['total_value']:,.2f}", delta=f"${pnl:+,.2f}")
        k2.metric("Bankroll (Cash)",       f"${snap['cash']:,.2f}")
        k3.metric("Holdings Value",        f"${snap['positions_value']:,.2f}")
        k4.metric("Open Positions",        str(len(snap["positions"])))
    else:
        k1, k2 = st.columns(2)
        k1.metric("Total Portfolio Value", f"${st.session_state['sim_capital']:,.2f}")
        k2.metric("Bankroll (Cash)",       f"${st.session_state['sim_capital']:,.2f}")

    st.divider()

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_floor, tab_portfolio, tab_performance, tab_market = st.tabs([
        "🎰  Trading Floor",
        "📊  Portfolio",
        "📈  Performance",
        "🌐  Market Data",
    ])

    daily_results    = st.session_state["daily_results"]
    strategy_lbl     = st.session_state["sim_strategy"]
    init_cap         = st.session_state["sim_capital"]
    portfolio_report = st.session_state.get("portfolio_report")

    with tab_floor:
        render_trading_floor(daily_results, strategy_lbl, portfolio_report)

    with tab_portfolio:
        render_portfolio(db, daily_results)

    with tab_performance:
        render_performance(db, init_cap, daily_results)

    with tab_market:
        render_market_data(db)


if __name__ == "__main__":
    main()

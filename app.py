"""
app.py  –  The Trading Floor
═════════════════════════════
Gambling-themed AI paper trading simulator.

Run:
    streamlit run app.py
"""

import logging
import random
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents import DataAgent, StrategyAgent, RiskAgent, ExecutionAgent

logger = logging.getLogger(__name__)
from config import (
    API_KEY,
    BACKTEST_DB_PATH,
    DB_PATH,
    INITIAL_CAPITAL,
    LOOKBACK_DAYS,
    MAX_POSITION_PCT,
    MAX_TRADES_PER_DAY,
    RESET_PASSWORD,
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
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Kill the sidebar entirely ── */
section[data-testid="stSidebar"]          { display: none !important; }
[data-testid="stSidebarCollapseButton"]   { display: none !important; }
[data-testid="collapsedControl"]          { display: none !important; }

/* ── Control strip separators ── */
.ctrl-top-border {
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, rgba(245,158,11,0.5) 20%,
        rgba(245,158,11,0.9) 50%, rgba(245,158,11,0.5) 80%, transparent 100%);
    margin: 0.6rem 0 0 0;
    border-radius: 1px;
}
.ctrl-bottom-border {
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%, rgba(245,158,11,0.25) 30%,
        rgba(245,158,11,0.25) 70%, transparent 100%);
    margin: 0.4rem 0 0.2rem 0;
}

/* ── Tabs — evenly spaced big buttons ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0 !important;
    border-bottom: 1px solid rgba(245,158,11,0.18) !important;
}
button[data-baseweb="tab"] {
    flex: 1 1 0 !important;
    justify-content: center !important;
    padding: 0.75rem 1rem !important;
    background: rgba(255,255,255,0.02) !important;
    border-radius: 0 !important;
    border-right: 1px solid rgba(245,158,11,0.10) !important;
    transition: background 0.15s !important;
}
button[data-baseweb="tab"]:hover {
    background: rgba(245,158,11,0.07) !important;
}
button[data-baseweb="tab"] p {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-size: 0.82rem !important;
    color: #9ca3af !important;
    text-align: center !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(245,158,11,0.10) !important;
    border-bottom: 3px solid #f59e0b !important;
}
[data-testid="stTabs"] [aria-selected="true"] p { color: #f59e0b !important; }

/* ── Dividers ── */
hr { border-color: rgba(245,158,11,0.18) !important; }

/* ── Captions / secondary text ── */
.stCaption { color: #6b7280 !important; font-size: 0.72rem !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid rgba(245,158,11,0.15) !important; }

/* ── Card containers (st.container border=True) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(245,158,11,0.22) !important;
    border-radius: 12px !important;
    padding: 0.65rem 0.8rem !important;
}

/* ── Equal-height side-by-side bordered cards ── */
[data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    height: 100% !important;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100% !important;
}

/* ── Hide Streamlit heading anchor links ── */
[data-testid="stHeadingAnchorLink"],
h1 a, h2 a, h3 a { display: none !important; }

/* ── Reduce gap between control strip cards and tabs ── */
.ctrl-bottom-border { margin: 0 !important; }
[data-testid="stTabs"] { margin-top: 0.25rem !important; }

/* ── st.metric styling — matches _html_metric look ── */
[data-testid="stMetric"] { text-align: center !important; }
[data-testid="stMetricLabel"] span {
    font-size: 0.65rem !important; font-weight: 800 !important;
    text-transform: uppercase !important; letter-spacing: 0.14em !important;
    color: #f59e0b !important;
}
[data-testid="stMetricValue"] {
    font-family: "Courier New", monospace !important;
    font-size: 1.65rem !important; font-weight: 900 !important; color: #f1f5f9 !important;
}
[data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"] svg { color: #9ca3af !important; }

/* ── Casino easter egg — popover looks like a plain emoji, not a button ── */
[data-testid="stPopover"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 0.15rem !important;
    min-height: unset !important;
    line-height: 1 !important;
    color: inherit !important;
}
[data-testid="stPopover"] button:hover,
[data-testid="stPopover"] button:focus {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stPopover"] button svg { display: none !important; }
[data-testid="stPopover"] button p {
    font-size: 1.8rem !important;
    line-height: 1 !important;
    margin: 0 !important;
}
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


def _html_metric(label: str, value: str, delta: str | None = None,
                 delta_positive: bool | None = None) -> str:
    """
    Return an HTML string for a metric card with inline delta.

    delta_positive:
        True  → green  (#10b981)
        False → red    (#ef4444)
        None  → gold   (#f59e0b)  — used when change = 0 or delta is omitted
    """
    if delta_positive is True:
        delta_color = "#10b981"
    elif delta_positive is False:
        delta_color = "#ef4444"
    else:
        delta_color = "#f59e0b"

    delta_html = ""
    if delta is not None:
        delta_html = (
            f"<span style='font-size:0.82rem; font-weight:700; color:{delta_color}; "
            f"margin-left:0.5rem; white-space:nowrap;'>{delta}</span>"
        )

    label_html = (
        f"<div style='font-size:0.62rem; font-weight:700; text-transform:uppercase; "
        f"letter-spacing:0.13em; color:#6b7280; margin-top:0.3rem;'>{label}</div>"
    ) if label else ""

    return (
        f"<div style='padding:0.55rem 0.1rem; text-align:center;'>"
        f"<div style='display:flex; align-items:baseline; flex-wrap:wrap; gap:0; justify-content:center;'>"
        f"<span style='font-family:\"Courier New\",monospace; font-size:1.65rem; "
        f"font-weight:900; color:#f1f5f9; line-height:1;'>{value}</span>"
        f"{delta_html}"
        f"</div>"
        f"{label_html}"
        f"</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading price data …")
def load_price_data(db_path: str) -> pd.DataFrame:
    """Load wide-format price history from the prices table of the given DB."""
    return PortfolioDatabase(db_path).load_prices()


def _sync_sim_prices(sim_db: "PortfolioDatabase") -> "pd.DataFrame":
    """
    Ensure sim_db (portfolio.db) has an up-to-date price history without
    downloading everything from scratch on every new game.

    Priority:
      1. sim_db already current      → incremental top-up only (seconds)
      2. sim_db empty, backtest_db has data → copy + incremental top-up
      3. Both empty                  → full yfinance fetch from 2019 (one-time)

    Returns the full prices DataFrame stored in sim_db.
    """
    from datetime import timedelta as _td
    _, sim_max = sim_db.get_prices_date_range()
    today = date.today()

    if sim_max is not None:
        # Prices exist — just refresh any missing recent trading days
        if sim_max.date() < today - _td(days=1):
            return md.refresh_latest_prices(sim_db)
        return sim_db.load_prices()

    # sim_db is empty — try to seed it from the backtest DB first
    bt_db = PortfolioDatabase(BACKTEST_DB_PATH)
    if bt_db.has_prices():
        bt_prices = bt_db.load_prices()
        if not bt_prices.empty:
            logger.info(
                "_sync_sim_prices: seeding sim DB from backtest DB "
                "(%d rows, %d tickers).",
                len(bt_prices) * len(bt_prices.columns),
                len(bt_prices.columns),
            )
            sim_db.upsert_prices(bt_prices)
            # Top-up with any dates newer than what the backtest DB had
            return md.refresh_latest_prices(sim_db)

    # Nothing anywhere — full historical fetch (only happens once ever)
    logger.info("_sync_sim_prices: no cached prices found — fetching from 2019.")
    return md.fetch_and_store_prices(sim_db, start="2019-01-01")


def _ensure_price_history() -> None:
    """
    Called during the loading screen on every server start.
    Uses _sync_sim_prices() so we never re-download data we already have.
    """
    sim_db = PortfolioDatabase(DB_PATH)
    try:
        _sync_sim_prices(sim_db)
    except Exception as exc:
        logger.warning("Price history sync failed: %s", exc)


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
# Trade Proposals Dialog
# ═══════════════════════════════════════════════════════════════════════════════

def show_proposals_dialog(db: "PortfolioDatabase", today_str: str) -> None:
    """
    Full-page trade confirmation screen — shown instead of the rest of the app
    whenever a proposal is pending.  Replaced @st.dialog for reliability.
    """
    st.markdown(
        "<h2 style='font-size:1.4rem;font-weight:900;text-transform:uppercase;"
        "letter-spacing:0.06em;color:#f1f5f9;margin-bottom:0.6rem;'>"
        "📋 Proposed Trades — Confirm to Execute</h2>",
        unsafe_allow_html=True,
    )
    data      = st.session_state["pending_day_data"]
    proposals = data["proposals"]
    analysis  = data["analysis"]
    portfolio = data["portfolio"]
    sim_date  = data["sim_date"]
    day_num   = data["day_num"]
    date_str  = sim_date.strftime("%Y-%m-%d")
    total_val = portfolio["total_value"]

    # ── API failure banner ────────────────────────────────────────────────────
    if data.get("api_failed"):
        st.error(
            f"**⚠️ Claude API unavailable** — trades were generated using the "
            f"deterministic fallback (same strategy logic, no AI reasoning).\n\n"
            f"**Error:** `{data.get('api_error', 'unknown error')}`",
            icon="🤖",
        )
    elif API_KEY:
        st.success("Trades proposed by Claude AI", icon="🤖")

    # ── Pre-validation risk summary ───────────────────────────────────────────
    risk_violations = data.get("risk_violations", [])
    if risk_violations:
        rejected = [v for v in risk_violations if v.startswith("REJECTED")]
        trimmed  = [v for v in risk_violations if v.startswith("TRIMMED")]
        if rejected:
            st.warning(
                f"**Risk Agent:** {len(rejected)} trade(s) rejected after {3} attempt(s). "
                "Showing best revised proposal.",
                icon="⚠️",
            )
        elif trimmed:
            st.info(
                f"**Risk Agent:** {len(trimmed)} trade(s) trimmed to stay within limits.",
                icon="✂️",
            )
        with st.expander(f"Risk Agent output ({len(risk_violations)} note(s))", expanded=False):
            for v in risk_violations:
                color = "#fca5a5" if v.startswith("REJECTED") else "#fde68a"
                st.markdown(
                    f"<div style='font-size:0.8rem; color:{color}; padding:2px 0;'>• {v}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        f"**Day {day_num}  ·  Market date: {date_str}**  "
        f"— Portfolio value: **${total_val:,.0f}**",
    )

    sells = [p for p in proposals if p["action"] == "SELL"]
    buys  = [p for p in proposals if p["action"] == "BUY"]

    # ── Liquidation block ─────────────────────────────────────────────────────
    if sells:
        st.markdown("#### 🔴  Liquidate positions")
        sell_rows = []
        for p in sells:
            pos   = portfolio.get("positions", {}).get(p["ticker"], {})
            price = analysis.get(p["ticker"], {}).get("price", 0.0)
            val   = pos.get("value", 0.0)
            pnl   = pos.get("pnl_pct", 0.0)
            sell_rows.append({
                "Ticker":          p["ticker"],
                "Current Price":   f"${price:,.2f}",
                "Position Value":  f"${val:,.0f}",
                "P&L":             f"{pnl:+.2f}%",
            })
        st.dataframe(
            sell_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No positions to liquidate — fully in cash.", icon="💵")

    # ── New buys block ────────────────────────────────────────────────────────
    if buys:
        import math as _math
        n_picks      = len(buys)
        alloc_pct    = round(100 / n_picks)
        # Mirror the risk-agent logic: equal split of (cash_after_sells × 0.995)
        # For display we approximate: total_val × 0.995 / n_picks
        per_stock_amt = total_val * 0.995 / n_picks
        st.markdown(f"#### 🟢  New positions (equal-weight, {alloc_pct}% each)")
        buy_rows = []
        for p in buys:
            m     = analysis.get(p["ticker"], {})
            price = m.get("price", 0.0)
            vol   = m.get("realized_vol_5d", 0.0)
            chg   = m.get("prev_day_return", 0.0)
            # Whole shares the execution agent will actually buy
            est_shares = _math.floor(per_stock_amt / price) if price > 0 else 0
            est_amt    = est_shares * price
            buy_rows.append({
                "Ticker":             p["ticker"],
                "Price":              f"${price:,.2f}",
                "5d Realised Vol":    f"{vol:.1%}",
                "Prev Day Change":    f"{chg:+.2f}%",
                "Est. Shares":        est_shares,
                "Amount to Invest":   f"${est_amt:,.0f}",
            })
        st.dataframe(
            buy_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Strategy: Top-50 S&P 500 stocks by 5-day realised volatility, "
            f"then the {n_picks} with the largest previous-day loss. "
            f"Each receives ~{alloc_pct}% of portfolio (${per_stock_amt:,.0f})."
        )
    else:
        st.warning("No buy candidates found in analysis data.", icon="⚠️")

    # ── Agent reasoning ───────────────────────────────────────────────────────
    reasoning = data.get("reasoning", "")
    if reasoning:
        st.caption(f"🧠 **Agent:** {reasoning}")

    st.divider()

    col_confirm, col_cancel = st.columns(2)

    with col_confirm:
        if st.button("✅  Execute Trades", type="primary", use_container_width=True):
            prices    = data["prices"]
            prices_df = load_price_data(db.db_path)

            risk_agent      = RiskAgent()
            portfolio_fresh = db.get_portfolio_state(prices)
            approved        = risk_agent.validate(proposals, portfolio_fresh, prices)

            execution_agent = ExecutionAgent(db)
            exec_report     = execution_agent.execute(approved, prices, date_str)

            result = {
                "date":            date_str,
                "day_num":         day_num,
                "portfolio":       exec_report["portfolio"],
                "analysis":        analysis,
                "proposed_trades": proposals,
                "approved_trades": exec_report["executed_trades"],
                "violations":      risk_agent.violations.copy(),
                "llm_reasoning":   data["reasoning"],
            }

            _log_agent({
                "day":     day_num,
                "date":    date_str,
                "agent":   "Execution Agent",
                "event":   "EXECUTE",
                "message": (
                    f"User confirmed. Executed {len(exec_report['executed_trades'])} trade(s). "
                    f"Portfolio: ${exec_report['equity']:,.0f}"
                ),
                "trades": [
                    {"ticker": t["ticker"], "action": t["action"],
                     "pct": t.get("pct_of_portfolio", 0),
                     "reasoning": t.get("reasoning", "")}
                    for t in exec_report["executed_trades"]
                ],
            })

            db.save_day_result(result)
            # Only gate on real calendar date in simulation mode
            if not data.get("backtest_date"):
                db.set_last_advance_date(today_str)
            st.session_state["daily_results"].append(result)
            st.session_state["sim_date_idx"] += 1
            st.session_state["sim_day_num"]  += 1
            st.session_state["report_date"]   = ""   # force report refresh
            st.session_state["pending_day_data"] = None
            st.rerun()

    with col_cancel:
        if st.button("❌  Cancel", use_container_width=True):
            _log_agent({
                "day":     day_num,
                "date":    date_str,
                "agent":   "User",
                "event":   "CANCEL",
                "message": "User cancelled the trade proposals — no trades executed.",
            })
            st.session_state["pending_day_data"] = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════════

_DARK = "plotly_dark"


def chart_equity_curve(history: pd.DataFrame, initial_capital: float) -> go.Figure:
    y_vals = history["total_equity"].tolist()
    y_min  = min(min(y_vals), initial_capital)
    y_max  = max(max(y_vals), initial_capital)
    span   = max(y_max - y_min, y_max * 0.004)   # at least 0.4 % of portfolio
    y_lo   = y_min - span * 0.35
    y_hi   = y_max + span * 0.35

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
        yaxis=dict(range=[y_lo, y_hi]),
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
# vs S&P 500 performance helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3_600, show_spinner=False)
def _load_spy_prices(start: str, end: str) -> pd.Series:
    """Fetch SPY adjusted close prices for [start-10d, end+3d]; empty on failure."""
    import yfinance as yf
    try:
        buf_s = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        buf_e = (pd.Timestamp(end)   + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        raw   = yf.download("SPY", start=buf_s, end=buf_e,
                             auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            return pd.Series(dtype=float, name="SPY")
        col = raw["Close"]
        return (col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col).squeeze().rename("SPY")
    except Exception:
        return pd.Series(dtype=float, name="SPY")


def _perf_metrics(
    port:    pd.Series,
    spy:     pd.Series,
    initial: float,
    rf_ann:  float = 0.05,
) -> dict:
    """
    Compute risk-adjusted performance metrics (portfolio vs SPY).
    Returns dict with 'ok' flag and all stat fields (NaN when < 2 data points).
    """
    TD  = 252
    rf  = (1 + rf_ann) ** (1 / TD) - 1
    nan = float("nan")

    spy_a = spy.reindex(port.index, method="ffill").dropna()
    idx   = port.index.intersection(spy_a.index)
    if len(idx) < 1:
        return {"ok": False}

    p, s       = port.loc[idx], spy_a.loc[idx]
    port_total = (p.iloc[-1] / initial  - 1) * 100
    spy_total  = (s.iloc[-1] / s.iloc[0] - 1) * 100

    pr_raw = p.pct_change().dropna()
    sr_raw = s.pct_change().dropna()
    n      = min(len(pr_raw), len(sr_raw))

    base = {"ok": True, "n": n, "port_total": port_total, "spy_total": spy_total}
    _na  = dict.fromkeys([
        "port_ann", "spy_ann", "port_vol", "spy_vol",
        "sharpe", "spy_sharpe", "sortino",
        "beta", "alpha", "treynor",
        "max_dd", "spy_max_dd",
        "info_ratio", "win_rate", "calmar",
    ], nan)

    if n < 1:
        return {**base, **_na}

    pr = pd.Series(pr_raw.values[:n])
    sr = pd.Series(sr_raw.values[:n])

    def _sd(v): return float("nan") if abs(v) < 1e-12 or pd.isna(v) else v

    port_ann = ((1 + pr.mean()) ** TD - 1) * 100
    spy_ann  = ((1 + sr.mean()) ** TD - 1) * 100

    if n >= 2:
        port_vol   = pr.std(ddof=1) * TD**0.5 * 100
        spy_vol    = sr.std(ddof=1) * TD**0.5 * 100
        sharpe     = (pr - rf).mean() / (pr.std(ddof=1) + 1e-12) * TD**0.5
        spy_sharpe = (sr - rf).mean() / (sr.std(ddof=1) + 1e-12) * TD**0.5

        dn       = pr[pr < rf]
        dn_std   = dn.std(ddof=1) if len(dn) > 1 else (abs(dn.iloc[0]) if len(dn) == 1 else 1e-12)
        sortino  = (pr.mean() - rf) / (dn_std + 1e-12) * TD**0.5

        beta     = _sd(pr.cov(sr) / (sr.var(ddof=1) + 1e-12))
        alpha_d  = pr.mean() - rf - beta * (sr.mean() - rf) if not pd.isna(beta) else nan
        alpha    = ((1 + alpha_d) ** TD - 1) * 100 if not pd.isna(alpha_d) else nan
        treynor  = _sd((port_ann / 100 - rf_ann) / (abs(beta) + 1e-12)) if not pd.isna(beta) else nan

        cum     = (1 + pr).cumprod()
        max_dd  = ((cum - cum.cummax()) / (cum.cummax() + 1e-12)).min() * 100
        scum    = (1 + sr).cumprod()
        spy_mdd = ((scum - scum.cummax()) / (scum.cummax() + 1e-12)).min() * 100

        act        = pd.Series(pr.values - sr.values)
        info_ratio = act.mean() / (act.std(ddof=1) + 1e-12) * TD**0.5
        win_rate   = (pr > 0).mean() * 100
        calmar     = _sd(port_ann / (abs(max_dd) + 1e-12)) if not pd.isna(max_dd) else nan
    else:
        port_vol = spy_vol = sharpe = spy_sharpe = sortino = nan
        beta = alpha = treynor = max_dd = spy_mdd = nan
        info_ratio = win_rate = calmar = nan

    return {
        **base,
        "port_ann": port_ann, "spy_ann": spy_ann,
        "port_vol": port_vol, "spy_vol": spy_vol,
        "sharpe":   sharpe,   "spy_sharpe": spy_sharpe,
        "sortino":  sortino,
        "beta":     beta,     "alpha":    alpha,  "treynor": treynor,
        "max_dd":   max_dd,   "spy_max_dd": spy_mdd,
        "info_ratio": info_ratio, "win_rate": win_rate, "calmar": calmar,
    }


def _render_vs_sp500(daily_results: list[dict]) -> None:
    """Renders the portfolio-vs-S&P-500 performance comparison widget."""
    dates  = [pd.Timestamp(r["date"]) for r in daily_results]
    values = [r["portfolio"]["total_value"] for r in daily_results]
    port   = pd.Series(values, index=pd.DatetimeIndex(dates), name="Portfolio")

    spy_raw = _load_spy_prices(daily_results[0]["date"], daily_results[-1]["date"])

    # ── Section header ────────────────────────────────────────────────────────
    nd    = len(daily_results)
    d_rng = (f"{daily_results[0]['date']} → {daily_results[-1]['date']}"
             if nd > 1 else daily_results[0]["date"])
    st.markdown(
        f"<div style='font-size:0.65rem; font-weight:800; text-transform:uppercase; "
        f"letter-spacing:0.12em; color:#f59e0b; margin-bottom:0.4rem;'>"
        f"📊 Portfolio vs S&P 500 (SPY)  ·  {nd} Day{'s' if nd != 1 else ''}  ·  {d_rng}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Cumulative return chart ───────────────────────────────────────────────
    t0       = dates[0] - pd.Timedelta(days=1)
    port_ext = pd.Series(
        [INITIAL_CAPITAL] + values,
        index=pd.DatetimeIndex([t0] + dates),
        name="Portfolio",
    )
    port_norm = port_ext / INITIAL_CAPITAL * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_norm.index, y=port_norm.values,
        name="Portfolio", mode="lines+markers",
        line=dict(color="#f59e0b", width=2.5),
        marker=dict(size=7, color="#f59e0b"),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.06)",
    ))
    # SPY: normalise from the first actual simulation date (Day 1), not from the
    # artificial t0 anchor.  This ensures both lines start at 100 on Day 1.
    spy_a = pd.Series(dtype=float)
    if not spy_raw.empty:
        _spy_days = spy_raw.reindex(pd.DatetimeIndex(dates), method="ffill").dropna()
        if len(_spy_days) >= 1:
            _spy_norm = _spy_days / _spy_days.iloc[0] * 100
            # Prepend t0 at 100 to align with the portfolio's visual anchor
            spy_a = pd.concat([
                pd.Series([100.0], index=pd.DatetimeIndex([t0])),
                _spy_norm,
            ])
            fig.add_trace(go.Scatter(
                x=spy_a.index, y=spy_a.values,
                name="S&P 500 (SPY)", mode="lines+markers",
                line=dict(color="#60a5fa", width=2, dash="dot"),
                marker=dict(size=5, color="#60a5fa"),
            ))
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(107,114,128,0.35)")

    # Collect all y-values from both traces to compute tight y-axis range
    _all_y = list(port_norm.values)
    if not spy_a.empty:
        _all_y += list(spy_a.values)
    _y_min = min(_all_y)
    _y_max = max(_all_y)
    _span  = max(_y_max - _y_min, 0.4)   # at least 0.4 index points
    _y_lo  = _y_min - _span * 0.35
    _y_hi  = _y_max + _span * 0.35

    fig.update_layout(
        template=_DARK, height=190,
        margin=dict(l=0, r=0, t=6, b=0),
        yaxis=dict(title="Indexed (100 = start)", tickformat=".2f",
                   range=[_y_lo, _y_hi]),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.2, x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Compute metrics ───────────────────────────────────────────────────────
    m = _perf_metrics(port, spy_raw, INITIAL_CAPITAL)
    if not m.get("ok"):
        st.caption("⚠️  Could not load SPY data — check internet connection.")
        return

    # ── Metric card helpers ───────────────────────────────────────────────────
    def _fv(key, fmt=".2f", suffix=""):
        """Format a metric value; return 'N/A' for NaN."""
        v = m.get(key, float("nan"))
        return "N/A" if pd.isna(v) else f"{v:{fmt}}{suffix}"

    def _card(label, port_key, spy_key=None, spy_fixed=None,
              fmt=".2f", suffix="", higher_better=True):
        """
        Build a metric comparison card.
        spy_fixed  – use a hardcoded comparison value instead of a dict key.
        """
        pv = m.get(port_key, float("nan"))
        sv = (m.get(spy_key, float("nan")) if spy_key
              else (spy_fixed if spy_fixed is not None else float("nan")))

        ps = "N/A" if pd.isna(pv) else f"{pv:{fmt}}{suffix}"
        ss = ("N/A" if pd.isna(sv)
              else (f"{sv:{fmt}}{suffix}" if spy_key or spy_fixed is not None else "—"))

        # Arrow direction = beating SPY or not
        if not pd.isna(pv) and not pd.isna(sv):
            win   = pv > sv if higher_better else pv < sv
            arrow = "▲" if win else "▼"
        else:
            arrow = ""

        # Color = sign of the portfolio value (negative→red, positive→green, zero→yellow)
        if pd.isna(pv):
            color = "#6b7280"
        elif pv > 0:
            color = "#10b981"
        elif pv < 0:
            color = "#ef4444"
        else:
            color = "#f59e0b"

        return (
            f"<div style='background:rgba(255,255,255,0.03);"
            f"border:1px solid rgba(245,158,11,0.15);border-radius:8px;"
            f"padding:10px 12px;min-height:76px;'>"
            f"<div style='font-size:0.55rem;color:#6b7280;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin-bottom:4px;'>{label}</div>"
            f"<div style='font-size:1.05rem;font-weight:900;color:{color};"
            f"font-family:monospace;line-height:1.2;'>"
            f"{ps} <span style='font-size:0.7rem;'>{arrow}</span></div>"
            f"<div style='font-size:0.65rem;color:#6b7280;margin-top:3px;'>"
            f"SPY&nbsp;<span style='color:#60a5fa;'>{ss}</span></div>"
            f"</div>"
        )

    # ── Row 1: Returns & efficiency ratios ────────────────────────────────────
    r1 = st.columns(4)
    r1[0].markdown(_card("Total Return",   "port_total", "spy_total",
                          fmt="+.2f", suffix="%"), unsafe_allow_html=True)
    r1[1].markdown(_card("Ann. Return",    "port_ann",   "spy_ann",
                          fmt="+.2f", suffix="%"), unsafe_allow_html=True)
    r1[2].markdown(_card("Sharpe Ratio",   "sharpe",     "spy_sharpe",
                          fmt=".2f"),               unsafe_allow_html=True)
    r1[3].markdown(_card("Sortino Ratio",  "sortino",    None,
                          fmt=".2f"),               unsafe_allow_html=True)

    # ── Row 2: Risk metrics ───────────────────────────────────────────────────
    r2 = st.columns(4)
    r2[0].markdown(_card("Beta",            "beta",    spy_fixed=1.0,
                          fmt=".2f",  higher_better=False),       unsafe_allow_html=True)
    r2[1].markdown(_card("Alpha (Jensen)", "alpha",   None,
                          fmt="+.2f", suffix="%"),                unsafe_allow_html=True)
    r2[2].markdown(_card("Max Drawdown",   "max_dd",  "spy_max_dd",
                          fmt=".2f",  suffix="%", higher_better=True), unsafe_allow_html=True)
    r2[3].markdown(_card("Ann. Volatility","port_vol","spy_vol",
                          fmt=".2f",  suffix="%", higher_better=False), unsafe_allow_html=True)

    # ── Footer: secondary metrics ─────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;gap:20px;margin-top:6px;font-size:0.7rem;"
        f"color:#9ca3af;flex-wrap:wrap;align-items:center;'>"
        f"<span>📈 Treynor: <b style='color:#f1f5f9;'>{_fv('treynor','.3f')}</b></span>"
        f"<span>⚖️ Info Ratio: <b style='color:#f1f5f9;'>{_fv('info_ratio','.2f')}</b></span>"
        f"<span>🏆 Win Rate: <b style='color:#f1f5f9;'>{_fv('win_rate','.1f','%')}</b></span>"
        f"<span>📉 Calmar: <b style='color:#f1f5f9;'>{_fv('calmar','.2f')}</b></span>"
        f"<span style='margin-left:auto;font-size:0.58rem;color:#4b5563;'>"
        f"n = {m['n']} return day(s)  ·  rf = 5% annual</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab renders
# ═══════════════════════════════════════════════════════════════════════════════

def render_daily_summary(daily_results: list[dict], strategy_lbl: str, portfolio_report: dict | None):
    """Daily Summary tab — today's P/L header, AI summary and strategy agent side-by-side."""

    # ── Day 0: no simulation started — show welcome card only ────────────────
    if not daily_results:
        st.markdown(
            f"""
            <div style='text-align:center; padding: 1.2rem 2rem 1rem;'>
                <div style='font-size:3.2rem; margin-bottom:0.6rem;'>🎰</div>
                <div style='font-size:0.72rem; font-weight:800; text-transform:uppercase;
                            letter-spacing:0.16em; color:#f59e0b; margin-bottom:0.4rem;'>
                    Welcome to The Trading Floor
                </div>
                <div style='font-size:2.4rem; font-weight:900; font-family:monospace;
                            color:#f1f5f9; margin-bottom:0.35rem;'>
                    ${INITIAL_CAPITAL:,.0f}
                </div>
                <div style='font-size:0.9rem; color:#6b7280; max-width:420px;
                            margin:0 auto; line-height:1.5;'>
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

    day      = daily_results[-1]
    snap     = day["portfolio"]
    init_cap = st.session_state["sim_capital"]

    # ── Header: 4 KPI metrics ─────────────────────────────────────────────────
    if len(daily_results) == 1:
        today_pnl, today_pnl_pct = 0.0, 0.0
    else:
        prev_val      = daily_results[-2]["portfolio"]["total_value"]
        today_pnl     = snap["total_value"] - prev_val
        today_pnl_pct = (today_pnl / prev_val * 100) if prev_val > 0 else 0.0

    # ── Header KPI row — no labels, centered numbers ─────────────────────────
    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.markdown(_html_metric("Portfolio Value", f"${snap['total_value']:,.0f}"),
                    unsafe_allow_html=True)
    with h2:
        st.markdown(_html_metric("Positions Value", f"${snap['positions_value']:,.0f}"),
                    unsafe_allow_html=True)
    with h3:
        st.markdown(_html_metric("Cash", f"${snap['cash']:,.0f}"),
                    unsafe_allow_html=True)
    with h4:
        _today_rounded = round(today_pnl)
        _pos = True if _today_rounded > 0 else (False if _today_rounded < 0 else None)
        st.markdown(
            _html_metric("Today's Return", f"{today_pnl_pct:+.2f}%",
                         delta=f"${today_pnl:+,.0f}", delta_positive=_pos),
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Two-column: AI Market Summary | Strategy Agent's Play ─────────────────
    col_summary, col_strategy = st.columns(2, gap="medium")

    _HDR = (
        "font-size:0.62rem; font-weight:800; text-transform:uppercase; "
        "letter-spacing:0.14em; color:#f59e0b; margin-bottom:0.8rem;"
    )

    with col_summary:
        with st.container(border=True):
            st.markdown(f"<div style='{_HDR}'>📰 AI Market Summary</div>",
                        unsafe_allow_html=True)
            if portfolio_report and portfolio_report.get("summary"):
                # Escape bare $ signs so Streamlit never treats them as LaTeX
                # math delimiters (e.g. "$248,749 … $248k" renders as italic
                # math mode without this).  \$ is displayed as a plain $.
                _safe_summary = portfolio_report["summary"].replace("$", r"\$")
                st.markdown(_safe_summary)
            else:
                st.caption("No AI summary yet — run a simulation day to generate one.")

    with col_strategy:
        with st.container(border=True):
            st.markdown(f"<div style='{_HDR}'>🤖 Strategy Agent's Play</div>",
                        unsafe_allow_html=True)

            approved_keys = {(t["ticker"], t["action"]) for t in day["approved_trades"]}

            if not day["proposed_trades"]:
                st.info("No trades proposed today.")
            else:
                rows_html = ""
                for prop in day["proposed_trades"]:
                    key       = (prop["ticker"], prop["action"])
                    is_buy    = prop["action"] == "BUY"
                    icon      = "🟢" if is_buy else "🔴"
                    verdict   = "✅ executed" if key in approved_keys else "❌ blocked"
                    color     = "#10b981" if is_buy else "#ef4444"
                    reasoning = prop.get("reasoning", "")
                    rows_html += (
                        f"<div style='padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);'>"
                        f"{icon} <span style='color:{color}; font-weight:700;'>"
                        f"{prop['action']} {prop['ticker']}</span>"
                        f"&nbsp;&nbsp;<span style='color:#9ca3af; font-size:0.8rem;'>{verdict}</span>"
                        f"<div style='font-size:0.72rem; color:#6b7280; margin-top:2px;'>{reasoning}</div>"
                        f"</div>"
                    )
                st.markdown(rows_html, unsafe_allow_html=True)

            if day["violations"]:
                st.markdown(
                    "<div style='margin-top:0.8rem; font-size:0.72rem; font-weight:700; "
                    "color:#f59e0b;'>⚠️ Risk Agent — Blocked Plays</div>",
                    unsafe_allow_html=True,
                )
                for v in day["violations"]:
                    st.warning(v, icon="⚠️")

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


def render_portfolio(
    db:               "PortfolioDatabase",
    daily_results:    list[dict],
    initial_capital:  float,
    portfolio_report: dict | None,
):
    if not daily_results:
        st.info("Start a simulation to see your portfolio.")
        return

    latest = daily_results[-1]
    snap   = latest["portfolio"]

    # ── Portfolio header KPIs (inline deltas via custom HTML) ────────────────
    if portfolio_report:
        r = portfolio_report
        h1, h2, h3, h4 = st.columns(4)

        with h1:
            st.markdown(_html_metric("Portfolio Value", f"${r['portfolio_value']:,.0f}"),
                        unsafe_allow_html=True)
        with h2:
            _r2 = round(r['fiveday_pnl'])
            _pos2 = True if _r2 > 0 else (False if _r2 < 0 else None)
            st.markdown(
                _html_metric("5-Day P/L", f"{r['fiveday_pnl_pct']:+.2f}%",
                             delta=f"${r['fiveday_pnl']:+,.0f}", delta_positive=_pos2),
                unsafe_allow_html=True,
            )
        with h3:
            _r3 = round(r['alltime_pnl'])
            _pos3 = True if _r3 > 0 else (False if _r3 < 0 else None)
            st.markdown(
                _html_metric("All-Time P/L", f"{r['alltime_pnl_pct']:+.2f}%",
                             delta=f"${r['alltime_pnl']:+,.0f}", delta_positive=_pos3),
                unsafe_allow_html=True,
            )
        with h4:
            st.metric(
                label="Win Rate",
                value=f"{r['win_rate']:.0f}%",
                help="% of trading days the portfolio closed higher than the previous day.",
            )
    else:
        st.markdown(_html_metric("Portfolio Value", f"${snap['total_value']:,.0f}"),
                    unsafe_allow_html=True)

    st.divider()

    # ── USD Portfolio Value graph (equity curve) ──────────────────────────────
    history_df = db.get_portfolio_history()
    if not history_df.empty:
        if daily_results:
            sim_start = daily_results[0]["date"]
            history_df = history_df[history_df["date"] >= sim_start]
        st.plotly_chart(
            chart_equity_curve(history_df, initial_capital),
            use_container_width=True,
        )
        st.divider()

    # ── Portfolio vs S&P 500 ──────────────────────────────────────────────────
    if len(daily_results) >= 1:
        _render_vs_sp500(daily_results)
        st.divider()

    # ── Holdings ─────────────────────────────────────────────────────────────
    st.markdown("**Holdings**")
    if snap["positions"]:
        rows = []
        for t, p in snap["positions"].items():
            # Use prices rounded to 2 dp so P/L is consistent with what is displayed
            avg_r = round(p["average_cost"],  2)
            cur_r = round(p["current_price"], 2)
            pnl_val = round(p["shares"] * (cur_r - avg_r), 2)
            pnl_pct = round((cur_r / avg_r - 1.0) * 100.0, 2) if avg_r > 0 else 0.0
            rows.append({
                "Ticker":    t,
                "Qty":       round(p["shares"], 4),
                "Avg Cost":  f"${avg_r:,.2f}",
                "Last Px":   f"${cur_r:,.2f}",
                "Mkt Value": f"${p['value']:,.2f}",
                "P/L $":     f"${pnl_val:+,.2f}",
                "P/L %":     f"{pnl_pct:+.2f}%",
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

    st.divider()

    # ── Daily Snapshots ───────────────────────────────────────────────────────
    st.markdown("**Daily Snapshots**")
    snap_rows = []
    for i, d in enumerate(daily_results):
        d_snap  = d["portfolio"]
        prev_eq = daily_results[i-1]["portfolio"]["total_value"] if i > 0 else initial_capital
        day_pnl = d_snap["total_value"] - prev_eq
        snap_rows.append({
            "Date":     d["date"],
            "Cash":     f"${d_snap['cash']:,.2f}",
            "Holdings": f"${d_snap['positions_value']:,.2f}",
            "Total":    f"${d_snap['total_value']:,.2f}",
            "Day P/L":  f"${day_pnl:+,.2f}",
            "Trades":   len(d["approved_trades"]),
            "Notes":    ", ".join(
                            f"{t['action']} {t['ticker']}"
                            for t in d["approved_trades"]
                        ) or "—",
        })

    if snap_rows:
        def _color_snap_pnl(val):
            if not isinstance(val, str):
                return ""
            if val.startswith("+") or (len(val) > 1 and val[1] == "+"):
                return "color:#10b981; font-weight:700"
            if val.startswith("-") or (len(val) > 1 and val[1] == "-"):
                return "color:#ef4444; font-weight:700"
            return ""

        st.dataframe(
            pd.DataFrame(snap_rows).style.map(_color_snap_pnl, subset=["Day P/L"]),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("Run at least one day to see the snapshot table.")


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


def _render_market_data_table(df: pd.DataFrame, price_date: str, mode_label: str) -> None:
    """Shared table renderer used by both simulation and backtest market data views."""
    df = df.copy()
    if "change_pct" not in df.columns:
        if "open" in df.columns and "close" in df.columns:
            df["change_pct"] = ((df["close"] - df["open"]) / df["open"] * 100).round(2)
        else:
            df["change_pct"] = float("nan")

    total   = len(df)
    gainers = int((df["change_pct"] > 0).sum())
    losers  = int((df["change_pct"] < 0).sum())

    st.markdown(
        f"<div style='font-size:0.72rem; font-weight:800; text-transform:uppercase; "
        f"letter-spacing:0.12em; color:#f59e0b; margin-bottom:0.3rem;'>"
        f"{mode_label}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"As per Yahoo Finance on {price_date}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(_html_metric("Stocks",      str(total)),   unsafe_allow_html=True)
    with m2:
        st.markdown(_html_metric("Winners",     str(gainers)), unsafe_allow_html=True)
    with m3:
        st.markdown(_html_metric("Losers",      str(losers)),  unsafe_allow_html=True)
    with m4:
        st.markdown(_html_metric("Market Date", price_date),   unsafe_allow_html=True)

    st.divider()

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("Search ticker or company", placeholder="e.g. AAPL or Apple")
    with fc2:
        sectors = ["All Sectors"] + sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else ["All Sectors"]
        sector_filter = st.selectbox("Sector", sectors)
    with fc3:
        direction = st.selectbox("Direction", ["All", "Winners only", "Losers only"])

    filtered = df.copy()
    if search:
        q = search.upper()
        sym_match  = filtered["symbol"].str.upper().str.contains(q, na=False) if "symbol" in filtered.columns else pd.Series(False, index=filtered.index)
        name_match = filtered["name"].str.upper().str.contains(q, na=False)   if "name"   in filtered.columns else pd.Series(False, index=filtered.index)
        filtered = filtered[sym_match | name_match]
    if sector_filter != "All Sectors" and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"] == sector_filter]
    if direction == "Winners only":
        filtered = filtered[filtered["change_pct"] > 0]
    elif direction == "Losers only":
        filtered = filtered[filtered["change_pct"] < 0]

    st.caption(f"Showing {len(filtered)} of {total} stocks")

    show_cols = [c for c in ["symbol", "name", "sector", "open", "high", "low", "close", "change_pct", "volume"] if c in filtered.columns]
    display = filtered[show_cols].copy()
    col_map = {"symbol": "Symbol", "name": "Company", "sector": "Sector",
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "change_pct": "Change %", "volume": "Volume"}
    display.columns = [col_map.get(c, c) for c in show_cols]

    def _style_change(val):
        if pd.isna(val):
            return ""
        return "color:#10b981; font-weight:700" if val > 0 else "color:#ef4444; font-weight:700"

    fmt = {}
    for col in ["Open", "High", "Low", "Close"]:
        if col in display.columns:
            fmt[col] = lambda v: f"${v:.2f}" if pd.notna(v) else "—"
    if "Change %" in display.columns:
        fmt["Change %"] = lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
    if "Volume" in display.columns:
        fmt["Volume"] = lambda v: f"{int(v):,}" if pd.notna(v) else "—"

    styled = display.style
    if "Change %" in display.columns:
        styled = styled.map(_style_change, subset=["Change %"])
    st.dataframe(
        styled.format(fmt),
        hide_index=True, use_container_width=True, height=600,
    )


def render_market_data(db: PortfolioDatabase, bt_prices_df: "pd.DataFrame | None" = None,
                       bt_cutoff_date: "pd.Timestamp | None" = None) -> None:
    """
    Renders the Market Data tab.

    In simulation mode (bt_prices_df=None) data comes from the DB (sp500_stocks table).
    In backtest mode, bt_prices_df is the backtest DB's prices DataFrame filtered to
    dates <= bt_cutoff_date; we build a day-by-day selector from those trading days.
    """
    if bt_prices_df is not None and bt_cutoff_date is not None:
        # ── Backtest mode: build view from DB close prices ─────────────────────
        bt_dates_done = bt_prices_df.index[bt_prices_df.index <= bt_cutoff_date]
        if bt_dates_done.empty:
            st.info("No backtest market data yet — advance at least one day.")
            return

        sorted_dates = sorted(bt_dates_done, reverse=True)
        day_labels   = [f"Trading Day {i + 1}" for i in range(len(sorted_dates) - 1, -1, -1)]
        label_to_ts  = dict(zip(day_labels, sorted_dates))

        selected_label = st.selectbox(
            "Snapshot", options=day_labels, index=0, label_visibility="collapsed"
        )
        selected_ts = label_to_ts[selected_label]

        # Build a DataFrame of close prices for the selected day
        close_row = bt_prices_df.loc[selected_ts]
        df_bt = pd.DataFrame({
            "symbol": close_row.index,
            "close":  close_row.values,
        })
        # Compute 1-day change if previous day exists
        prev_dates = bt_prices_df.index[bt_prices_df.index < selected_ts]
        if not prev_dates.empty:
            prev_ts   = prev_dates[-1]
            prev_row  = bt_prices_df.loc[prev_ts]
            aligned   = prev_row.reindex(df_bt["symbol"])
            df_bt["open"] = aligned.values
        else:
            df_bt["open"] = df_bt["close"]

        df_bt = df_bt.dropna(subset=["close"])
        # Enrich with name/sector from the simulation DB's sp500_stocks table
        # (the backtest DB's prices table only stores date/symbol/close)
        try:
            _sim_stocks = PortfolioDatabase(DB_PATH).get_sp500_stocks()
            if not _sim_stocks.empty:
                _meta = (
                    _sim_stocks[["symbol", "name", "sector"]]
                    .drop_duplicates("symbol")
                )
                df_bt = df_bt.merge(_meta, on="symbol", how="left")
                df_bt["name"]   = df_bt["name"].fillna(df_bt["symbol"])
                df_bt["sector"] = df_bt["sector"].fillna("—")
            else:
                df_bt["name"]   = df_bt["symbol"]
                df_bt["sector"] = "—"
        except Exception:
            df_bt["name"]   = df_bt["symbol"]
            df_bt["sector"] = "—"

        price_date = selected_ts.strftime("%Y-%m-%d")
        _render_market_data_table(df_bt, price_date, "S&P 500 Historical Market Data (Backtest)")
        return

    # ── Simulation mode: read from DB ─────────────────────────────────────────
    fetch_dates = db.get_sp500_fetch_dates()

    if not fetch_dates:
        st.info("No market data yet — start a new game to fetch today's quotes.")
        return

    # Build "Trading Day N" labels (most recent = highest day number)
    day_labels = [f"Trading Day {len(fetch_dates) - i}" for i in range(len(fetch_dates))]
    label_to_date = dict(zip(day_labels, fetch_dates))

    selected_label = st.selectbox(
        "Snapshot", options=day_labels, index=0, label_visibility="collapsed"
    )
    selected_date = label_to_date[selected_label]

    df = db.get_sp500_stocks(selected_date)
    if df.empty:
        st.warning("No data for this date.")
        return

    # Derive the actual market date of the price data
    price_date = selected_date
    if "price_date" in df.columns:
        _pd_vals = df["price_date"].dropna()
        if not _pd_vals.empty:
            price_date = str(_pd_vals.iloc[0])

    _render_market_data_table(df, price_date, "S&P 500 Live Market Data")



# ═══════════════════════════════════════════════════════════════════════════════
# Agent Logs tab renderer
# ═══════════════════════════════════════════════════════════════════════════════

_AGENT_COLORS = {
    "Data Agent":      ("#3b82f6", "#1e3a5f"),
    "Strategy Agent":  ("#f59e0b", "#3d2e08"),
    "Risk Agent":      ("#ef4444", "#3d1515"),
    "Execution Agent": ("#10b981", "#0a2e21"),
    "User":            ("#8b5cf6", "#2a1f4a"),
}
_EVENT_BADGES = {
    "ANALYZE":  ("#3b82f6",  "ANALYZE"),
    "PROPOSE":  ("#f59e0b",  "PROPOSE"),
    "REVISE":   ("#f97316",  "REVISE"),
    "REJECT":   ("#ef4444",  "REJECT"),
    "APPROVE":  ("#10b981",  "APPROVE"),
    "EXECUTE":  ("#10b981",  "EXECUTE"),
    "CANCEL":   ("#6b7280",  "CANCEL"),
}


def render_agent_logs() -> None:
    """Render the agent pipeline log — table view + timeline view."""
    logs: list[dict] = st.session_state.get("agent_logs", [])

    st.markdown(
        "<div style='font-size:0.65rem; font-weight:800; text-transform:uppercase; "
        "letter-spacing:0.12em; color:#f59e0b; margin-bottom:0.75rem;'>"
        "🤖 Agent Pipeline — Event Log</div>",
        unsafe_allow_html=True,
    )

    if not logs:
        st.info("No agent events yet — deal a hand to see the pipeline in action.", icon="🤖")
        return

    tab_table, tab_timeline = st.tabs(["📋  Table", "🕐  Timeline"])

    # ── TABLE VIEW ────────────────────────────────────────────────────────────
    with tab_table:
        rows = []
        for e in logs:
            rows.append({
                "Day":     e.get("day", ""),
                "Date":    e.get("date", ""),
                "Time":    e.get("ts", ""),
                "Agent":   e.get("agent", ""),
                "Event":   e.get("event", ""),
                "Message": e.get("message", ""),
                "Trades":  len(e.get("trades", [])),
                "Violations": len(e.get("violations", [])),
            })
        df_logs = pd.DataFrame(rows)

        def _style_agent(val):
            color = _AGENT_COLORS.get(val, ("#9ca3af", "#1f2937"))[0]
            return f"color:{color}; font-weight:700"

        def _style_event(val):
            color = _EVENT_BADGES.get(val, ("#9ca3af", val))[0]
            return f"color:{color}; font-weight:800"

        def _style_violations(val):
            if isinstance(val, int) and val > 0:
                return "color:#ef4444; font-weight:700"
            return "color:#6b7280"

        st.dataframe(
            df_logs.style
            .map(_style_agent, subset=["Agent"])
            .map(_style_event, subset=["Event"])
            .map(_style_violations, subset=["Violations"]),
            hide_index=True,
            use_container_width=True,
            height=min(400, 40 + len(rows) * 36),
        )

        st.divider()
        if st.button("🗑️  Clear Log", key="clear_log_table"):
            try:
                PortfolioDatabase().reset.__func__  # just check db is accessible
            except Exception:
                pass
            # Clear from DB by running a raw delete
            try:
                db_instance = PortfolioDatabase()
                with db_instance._conn() as c:
                    c.execute("DELETE FROM agent_logs")
            except Exception:
                pass
            st.session_state["agent_logs"] = []
            st.rerun()

    # ── TIMELINE VIEW ─────────────────────────────────────────────────────────
    with tab_timeline:
        from itertools import groupby
        days = []
        for day_key, group in groupby(logs, key=lambda e: (e.get("day", "?"), e.get("date", ""))):
            days.append((day_key, list(group)))

        for (day_num, date_str), events in reversed(days):
            latest_day = days[-1][0][0]
            with st.expander(
                f"Day {day_num}  ·  {date_str}  ·  {len(events)} event(s)",
                expanded=(day_num == latest_day),
            ):
                for entry in events:
                    agent  = entry.get("agent", "Unknown")
                    event  = entry.get("event", "")
                    msg    = entry.get("message", "")
                    ts     = entry.get("ts", "")

                    ac, abg = _AGENT_COLORS.get(agent, ("#9ca3af", "#1f2937"))
                    ec, elabel = _EVENT_BADGES.get(event, ("#9ca3af", event))

                    st.markdown(
                        f"<div style='display:flex; gap:0.75rem; align-items:flex-start; "
                        f"padding:0.65rem 0; border-bottom:1px solid rgba(255,255,255,0.05);'>"
                        f"<div style='min-width:110px; text-align:right;'>"
                        f"<div style='font-size:0.6rem; color:#6b7280; font-family:monospace;'>{ts}</div>"
                        f"<div style='display:inline-block; margin-top:3px; padding:2px 7px; "
                        f"background:{abg}; border:1px solid {ac}44; border-radius:4px; "
                        f"font-size:0.6rem; font-weight:700; color:{ac}; text-transform:uppercase; "
                        f"letter-spacing:0.08em; white-space:nowrap;'>{agent}</div>"
                        f"</div>"
                        f"<div style='flex:1;'>"
                        f"<span style='display:inline-block; padding:2px 8px; background:{ec}22; "
                        f"border:1px solid {ec}55; border-radius:4px; font-size:0.6rem; "
                        f"font-weight:800; color:{ec}; text-transform:uppercase; "
                        f"letter-spacing:0.1em; margin-right:8px;'>{elabel}</span>"
                        f"<span style='font-size:0.82rem; color:#d1d5db;'>{msg}</span>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    trades = entry.get("trades", [])
                    if trades:
                        with st.expander(f"  ↳ {len(trades)} trade(s)", expanded=False):
                            st.dataframe(
                                [{"Action": t.get("action",""), "Ticker": t.get("ticker",""),
                                  "Alloc %": f"{t.get('pct',0):.1f}%",
                                  "Reasoning": t.get("reasoning","")} for t in trades],
                                hide_index=True, use_container_width=True,
                            )

                    violations = entry.get("violations", [])
                    if violations:
                        v_html = "".join(
                            f"<div style='font-size:0.72rem;color:#fca5a5;padding:2px 0;'>⚠️ {v}</div>"
                            for v in violations
                        )
                        st.markdown(
                            f"<div style='margin:4px 0 4px 120px;background:rgba(239,68,68,0.07);"
                            f"border-left:2px solid #ef4444;border-radius:0 4px 4px 0;"
                            f"padding:6px 10px;'>{v_html}</div>",
                            unsafe_allow_html=True,
                        )

        st.divider()
        if st.button("🗑️  Clear Log", key="clear_log_timeline"):
            try:
                db_instance = PortfolioDatabase()
                with db_instance._conn() as c:
                    c.execute("DELETE FROM agent_logs")
            except Exception:
                pass
            st.session_state["agent_logs"] = []
            st.rerun()


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
# Deal-next-hand logic (shared by button click and auto-deal after Start Game)
# ═══════════════════════════════════════════════════════════════════════════════

def _log_agent(entry: dict) -> None:
    """Append a structured log entry to session state and persist to DB."""
    import datetime as _dt
    entry.setdefault("ts", _dt.datetime.now().strftime("%H:%M:%S"))
    st.session_state.setdefault("agent_logs", []).append(entry)
    try:
        _db_path = (BACKTEST_DB_PATH
                    if st.session_state.get("app_mode") == "backtest"
                    else DB_PATH)
        PortfolioDatabase(_db_path).save_agent_log(entry)
    except Exception:
        pass  # never let logging break the app


def _deal_next_hand(
    db: "PortfolioDatabase",
    backtest_date: "pd.Timestamp | None" = None,
) -> None:
    """
    Run the full proposal pipeline for the next simulation day.

    backtest_date – when set (backtest mode), use this historical date directly
                    and skip the live market-data refresh step.

    Pipeline:
      DataAgent → StrategyAgent → RiskAgent (pre-validate)
        └─ if violations → StrategyAgent (retry with feedback, up to 2 retries)
      Stores result in pending_day_data; dialog fires on next rerun.
    """
    day_num      = st.session_state["sim_day_num"]
    strategy_now = st.session_state["sim_strategy"]

    # ── 1. Fetch latest market data (simulation only) ─────────────────────────
    if backtest_date is None:
        with st.spinner("Fetching latest market data …"):
            try:
                md.refresh_latest_prices(db)
                md.fetch_and_store_sp500(db)
                st.cache_data.clear()
            except Exception as exc:
                st.warning(f"Could not refresh market data: {exc}")
        prices_df_fresh = load_price_data(db.db_path)
        sim_date        = prices_df_fresh.index.max()
    else:
        # Backtest: prices already scoped to the backtest window in backtest DB
        prices_df_fresh = load_price_data(db.db_path)
        sim_date        = pd.Timestamp(backtest_date)

    date_str = sim_date.strftime("%Y-%m-%d")
    # Guard: if sim_date is not in the loaded prices index (e.g. due to NaN-only
    # rows being skipped during upsert), fall back to the nearest available date.
    if sim_date not in prices_df_fresh.index:
        _avail = prices_df_fresh.index[prices_df_fresh.index <= sim_date]
        sim_date = _avail[-1] if not _avail.empty else prices_df_fresh.index[0]
        date_str = sim_date.strftime("%Y-%m-%d")
    prices          = prices_df_fresh.loc[sim_date].to_dict()
    portfolio       = db.get_portfolio_state(prices)

    _log_agent({"day": day_num, "date": date_str, "agent": "Data Agent",
                "event": "ANALYZE",
                "message": f"Analysed {len(prices)} tickers for {date_str}."})

    # ── 2. Strategy → Risk feedback loop (max 3 attempts) ────────────────────
    _spinner_base = "Asking Claude" if API_KEY else "Running deterministic strategy"
    data_agent     = DataAgent()
    strategy_agent = StrategyAgent(strategy_now)
    analysis       = data_agent.analyze(prices_df_fresh, sim_date)

    feedback   : list[str] | None = None
    proposed   : list[dict] = []
    risk_violations: list[str] = []
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        attempt_label = f"attempt {attempt + 1}/{MAX_RETRIES}" if attempt > 0 else ""
        spinner_msg   = (
            f"{_spinner_base} — Day {day_num} trades"
            + (f" ({attempt_label}, revising after risk feedback)" if attempt_label else " …")
        )
        with st.spinner(spinner_msg):
            proposed = strategy_agent.propose_trades(
                analysis, portfolio, date_str, day_num, feedback=feedback
            )

        n_sell = sum(1 for p in proposed if p["action"] == "SELL")
        n_buy  = sum(1 for p in proposed if p["action"] == "BUY")
        _log_agent({
            "day": day_num, "date": date_str,
            "agent": "Strategy Agent",
            "event": "PROPOSE" if attempt == 0 else "REVISE",
            "message": (
                f"Proposed {n_sell} SELL(s) + {n_buy} BUY(s)."
                + (f" [Revision {attempt}]" if attempt > 0 else "")
            ),
            "trades": [{"ticker": p["ticker"], "action": p["action"],
                        "pct": p.get("pct_of_portfolio", 0),
                        "reasoning": p.get("reasoning", "")} for p in proposed],
            "reasoning": strategy_agent.reasoning,
            "api_failed": strategy_agent.api_failed,
        })

        # Pre-validate with RiskAgent
        risk_agent = RiskAgent()
        risk_agent.validate(proposed, portfolio, prices)
        risk_violations = risk_agent.violations.copy()

        rejected = [v for v in risk_violations if v.startswith("REJECTED")]

        if rejected:
            _log_agent({
                "day": day_num, "date": date_str,
                "agent": "Risk Agent",
                "event": "REJECT",
                "message": f"{len(rejected)} violation(s) on attempt {attempt + 1}.",
                "violations": risk_violations,
            })
            if attempt < MAX_RETRIES - 1:
                feedback = risk_violations   # pass to next strategy iteration
                continue
        else:
            _log_agent({
                "day": day_num, "date": date_str,
                "agent": "Risk Agent",
                "event": "APPROVE",
                "message": (
                    "All proposed trades passed risk constraints."
                    + (f" {len(risk_violations)} trim(s) applied."
                       if risk_violations else "")
                ),
                "violations": risk_violations,
            })
        break   # no hard rejections — proceed

    # ── 3. Store payload; dialog fires on next rerun ──────────────────────────
    st.session_state["pending_day_data"] = {
        "proposals":       proposed,
        "analysis":        analysis,
        "sim_date":        sim_date,
        "prices":          prices,
        "strategy":        strategy_now,
        "reasoning":       strategy_agent.reasoning,
        "day_num":         day_num,
        "portfolio":       portfolio,
        "api_failed":      strategy_agent.api_failed,
        "api_error":       strategy_agent.api_error,
        "risk_violations": risk_violations,
        # backtest_date is set in backtest mode so the dialog skips the advance-date gate
        "backtest_date":   date_str if backtest_date is not None else None,
    }
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Session restore helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _restore_session_from_db(db: PortfolioDatabase) -> None:
    """
    On the first Streamlit page load of a new browser session, restore
    simulation state from SQLite so the user does not lose progress after a
    page refresh or server restart.

    Only runs once per session (guard: checks for 'sim_started' in session
    state which is always set by the defaults loop that follows this call).
    """
    if "sim_started" in st.session_state:
        return  # already initialised this session — nothing to do

    if not db.has_simulation():
        return  # DB was never seeded — let the user click Start New Game

    results = db.load_all_day_results()

    # Mark the simulation as active regardless of whether any days are done
    st.session_state["sim_started"]      = True
    st.session_state["daily_results"]    = results
    st.session_state["sim_date_idx"]     = len(results)
    st.session_state["sim_day_num"]      = len(results) + 1
    st.session_state["sim_dates"]        = []          # rebuilt on each Next Hand
    st.session_state["sim_capital"]      = INITIAL_CAPITAL
    st.session_state["sim_strategy"]     = "Momentum"
    st.session_state["pending_day_data"] = None
    st.session_state["agent_logs"]       = db.load_agent_logs()


# ═══════════════════════════════════════════════════════════════════════════════
# Loading screen
# ═══════════════════════════════════════════════════════════════════════════════

def _show_loading_screen() -> None:
    """Animated full-page loading splash shown once on cold app start."""
    st.markdown(
        _LOADING_SCREEN_CSS + """
<div style="min-height:100vh;width:100%;background:#0a0a0a;
            display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:1.6rem;">
  <div style="font-size:5.5rem;animation:ld-spin 1.4s linear infinite;
              display:inline-block;filter:drop-shadow(0 0 20px rgba(245,158,11,0.4));">🎰</div>
  <div style="text-align:center;">
    <h1 style="font-weight:900;color:#f1f5f9;text-transform:uppercase;
               letter-spacing:0.08em;font-size:1.85rem;margin:0 0 0.5rem;">
      AI Investment Application
    </h1>
    <div style="color:#f59e0b;font-size:0.75rem;font-weight:800;
                text-transform:uppercase;letter-spacing:0.28em;">
      Initialising market engine&hellip;
    </div>
  </div>
  <div style="display:flex;gap:7px;margin-top:0.2rem;">
    <span class="ld-dot"></span>
    <span class="ld-dot" style="animation-delay:.18s"></span>
    <span class="ld-dot" style="animation-delay:.36s"></span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    time.sleep(1.8)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# New-game onboarding screens
# ═══════════════════════════════════════════════════════════════════════════════

def _render_mode_select() -> None:
    """Full-page mode selection screen — shown on fresh load and 'Start New Game'."""
    st.markdown("""
<style>
header,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{padding:0!important;max-width:860px!important;margin:0 auto!important;}
@keyframes ms-fade{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
</style>
""", unsafe_allow_html=True)

    _err = st.session_state.pop("_new_game_error", None)
    if _err:
        st.error(f"Error: {_err}")

    # Check if an active simulation exists (backtest always starts fresh).
    _sim_db_check   = PortfolioDatabase(DB_PATH)
    _sim_has_trades = bool(_sim_db_check.load_all_day_results())

    st.markdown("""
<div style="padding:2vh 1.5rem 1.2rem;text-align:center;animation:ms-fade 0.55s ease-out;">
  <div style="font-size:4.5rem;margin-bottom:0.6rem;">🎰</div>
  <div style="font-size:0.68rem;font-weight:800;text-transform:uppercase;
              letter-spacing:0.4em;color:#f59e0b;margin-bottom:0.4rem;">Select Mode</div>
  <h1 style="font-size:2.6rem;font-weight:900;color:#f1f5f9;text-transform:uppercase;
             letter-spacing:0.05em;margin:0 0 0.3rem;line-height:1.1;">
    How Do You Want to Play?
  </h1>
  <p style="color:#6b7280;font-size:0.85rem;margin:0 0 1.2rem;letter-spacing:0.06em;">
    Choose your trading mode to get started
  </p>
</div>
""", unsafe_allow_html=True)

    col_sim, col_bt = st.columns(2, gap="large")

    with col_sim:
        with st.container(border=True):
            st.markdown("""
<div style="text-align:center;padding:1rem 1rem 0.5rem;">
  <div style="font-size:3rem;margin-bottom:0.6rem;">📈</div>
  <div style="font-size:1.15rem;font-weight:900;color:#f1f5f9;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:0.7rem;">Live Simulation</div>
  <div style="font-size:0.8rem;color:#9ca3af;line-height:1.7;margin-bottom:1.2rem;">
    Trade in real time.<br>
    Advance one day per 24 hours.<br>
    Uses today's live market data.
  </div>
  <div style="font-size:0.62rem;color:#10b981;letter-spacing:0.06em;
              line-height:1.9;margin-bottom:1.2rem;">
    ✓ Live S&amp;P 500 data &nbsp;·&nbsp; ✓ Claude AI strategy<br>
    ✓ 14-day game (~10 trades) &nbsp;·&nbsp; ✓ vs S&amp;P 500
  </div>
</div>""", unsafe_allow_html=True)
            # Show "Continue" as primary CTA if an active simulation exists.
            if _sim_has_trades:
                if st.button("▶  Continue Simulation", type="primary", use_container_width=True):
                    st.session_state["app_mode"] = "simulation"
                    st.session_state["new_game_stage"] = None
                    st.rerun()
                if st.button("📈  New Simulation", use_container_width=True):
                    st.session_state["app_mode"] = "simulation"
                    st.session_state["new_game_stage"] = "resetting"
                    st.rerun()
            else:
                if st.button("📈  Start Simulation", type="primary", use_container_width=True):
                    st.session_state["app_mode"] = "simulation"
                    st.session_state["new_game_stage"] = "resetting"
                    st.rerun()

    with col_bt:
        with st.container(border=True):
            st.markdown("""
<div style="text-align:center;padding:1rem 1rem 0.5rem;">
  <div style="font-size:3rem;margin-bottom:0.6rem;">⏱️</div>
  <div style="font-size:1.15rem;font-weight:900;color:#f1f5f9;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:0.7rem;">Back Testing</div>
  <div style="font-size:0.8rem;color:#9ca3af;line-height:1.7;margin-bottom:1.2rem;">
    Replay historical market data.<br>
    Advance at your own pace.<br>
    Test strategy on the past.
  </div>
  <div style="font-size:0.62rem;color:#60a5fa;letter-spacing:0.06em;
              line-height:1.9;margin-bottom:1.2rem;">
    ✓ History from 2020 &nbsp;·&nbsp; ✓ No time restrictions<br>
    ✓ 14-day window &nbsp;·&nbsp; ✓ Strategy validation
  </div>
</div>""", unsafe_allow_html=True)
            # Backtest always starts fresh — no continue option.
            if st.button("⏱️  Start Back Testing", use_container_width=True):
                st.session_state["app_mode"] = "backtest"
                st.session_state["new_game_stage"] = "bt_setup"
                st.rerun()


def _render_bt_setup() -> None:
    """Back testing date-selection screen."""
    st.markdown("""
<style>
header,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{padding:0!important;max-width:620px!important;margin:0 auto!important;}
@keyframes bs-fade{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
.stButton button[kind="primary"]{
  background:linear-gradient(135deg,#3b82f6 0%,#2563eb 100%)!important;
  color:#fff!important;font-weight:900!important;}
</style>
""", unsafe_allow_html=True)

    _err = st.session_state.pop("_new_game_error", None)
    if _err:
        st.error(f"Setup failed: {_err} — please try again.")

    st.markdown("""
<div style="padding:2vh 1.5rem 1rem;text-align:center;animation:bs-fade 0.5s ease-out;">
  <div style="font-size:3.5rem;margin-bottom:0.5rem;">⏱️</div>
  <div style="font-size:0.68rem;font-weight:800;text-transform:uppercase;
              letter-spacing:0.4em;color:#3b82f6;margin-bottom:0.4rem;">Configure</div>
  <h1 style="font-size:2.2rem;font-weight:900;color:#f1f5f9;text-transform:uppercase;
             letter-spacing:0.05em;margin:0 0 0.3rem;">Back Testing Setup</h1>
  <p style="color:#6b7280;font-size:0.85rem;margin:0 0 1rem;letter-spacing:0.05em;">
    Pick a start date — the AI will replay the strategy day by day from there
  </p>
</div>
""", unsafe_allow_html=True)

    sim_db    = PortfolioDatabase(DB_PATH)
    all_dates = sim_db.get_prices_dates()  # fast: distinct dates only, no price data loaded

    from datetime import timedelta as _td, date as _date
    _BT_WINDOW = 14  # calendar days per backtest game

    # Picker always spans 2020-01-01 → today-15d.
    # Preview card shows 0 trading days if the selected range has no data yet.
    min_date     = _date(2020, 1, 1)
    max_date     = _date.today() - _td(days=15)
    default_date = max(min_date, max_date - _td(days=30))

    _, date_col, _ = st.columns([1, 3, 1])
    with date_col:
        st.markdown(
            "<div style='font-size:0.78rem;font-weight:700;color:#94a3b8;"
            "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;"
            "text-align:center;'>"
            "📅 Backtest Start Date</div>",
            unsafe_allow_html=True,
        )
        _yr_col, _mo_col, _dy_col = st.columns([1, 1, 1])

        _years  = list(range(min_date.year, max_date.year + 1))
        _months = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

        # ── Initialise / clamp the picker state ──────────────────────────────
        _ss_y = st.session_state.get("_bt_pick_year",  default_date.year)
        _ss_m = st.session_state.get("_bt_pick_month", default_date.month)
        _ss_d = st.session_state.get("_bt_pick_day",   default_date.day)

        with _yr_col:
            _sel_year = st.selectbox(
                "Year", options=_years,
                index=_years.index(_ss_y) if _ss_y in _years else len(_years) - 1,
                label_visibility="visible", key="_bt_pick_year",
            )
        with _mo_col:
            _sel_month = st.selectbox(
                "Month", options=list(range(1, 13)),
                index=_ss_m - 1,
                format_func=lambda m: _months[m - 1],
                label_visibility="visible", key="_bt_pick_month",
            )
        with _dy_col:
            import calendar as _cal
            _max_day_in_month = _cal.monthrange(_sel_year, _sel_month)[1]
            _day_opts = list(range(1, _max_day_in_month + 1))
            _clamped_day = min(_ss_d, _max_day_in_month)
            _sel_day = st.selectbox(
                "Day", options=_day_opts,
                index=_clamped_day - 1,
                label_visibility="visible", key="_bt_pick_day",
            )

        # Clamp to allowed range
        import datetime as _dt_mod
        _raw_date = _dt_mod.date(_sel_year, _sel_month, _sel_day)
        selected_date = max(min_date, min(max_date, _raw_date))

        if selected_date != _raw_date:
            st.caption(f"⚠️ Clamped to allowed range: {selected_date}")

        # Pick the first SIMULATION_DAYS trading days from selected start
        _start_ts         = pd.Timestamp(selected_date)
        _trading_from_start = sorted([d for d in all_dates if d >= _start_ts])
        bt_dates          = _trading_from_start[:SIMULATION_DAYS]
        n_days            = len(bt_dates)
        end_label         = bt_dates[-1].strftime("%b %d, %Y") if bt_dates else (
            _start_ts + pd.Timedelta(days=_BT_WINDOW - 1)).strftime("%b %d, %Y")

        _no_data = n_days == 0
        if _no_data:
            # No prices cached yet for this window — will be fetched on start
            st.markdown(f"""
<div style="background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.18);
            border-radius:10px;padding:1rem 1.2rem;text-align:center;margin:1rem 0;">
  <div style="font-size:0.75rem;font-weight:700;color:#60a5fa;margin-bottom:4px;">
    ⏳ Price data will be fetched on start
  </div>
  <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;
              color:#4b5563;margin-top:3px;">This takes ~1 min the first time</div>
  <div style="font-size:0.65rem;color:#374151;margin-top:8px;">
    {SIMULATION_DAYS} trading days from {selected_date.strftime("%b %d, %Y")}
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.22);
            border-radius:10px;padding:1rem 1.2rem;text-align:center;margin:1rem 0;">
  <div style="font-size:2rem;font-weight:900;font-family:monospace;color:#60a5fa;">{n_days}</div>
  <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;
              color:#6b7280;margin-top:3px;">Trading days in this window</div>
  <div style="font-size:0.65rem;color:#4b5563;margin-top:8px;">
    {selected_date.strftime("%b %d, %Y")} → {end_label}
  </div>
</div>""", unsafe_allow_html=True)

        if st.button("⏱️  Begin Back Testing", type="primary", use_container_width=True):
            st.session_state["bt_start_date"] = selected_date.isoformat()
            st.session_state["new_game_stage"] = "bt_resetting"
            st.rerun()

        st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
        if st.button("← Back to Mode Select", use_container_width=True):
            st.session_state["new_game_stage"] = "mode_select"
            st.rerun()


def _start_backtest_with_loading(db: "PortfolioDatabase") -> None:
    """Reset the backtest DB and configure session state for back-testing."""

    _BT_WINDOW     = 21  # calendar-day ceiling — guarantees 10+ trading days
    start_date_str = st.session_state.get("bt_start_date", "")
    sim_db         = PortfolioDatabase(DB_PATH)

    # ── Validate start date ──────────────────────────────────────────────────
    start_ts = pd.Timestamp(start_date_str) if start_date_str else None
    if start_ts is None:
        st.session_state["new_game_stage"]  = "bt_setup"
        st.session_state["_new_game_error"] = "No backtest start date set."
        st.rerun()
        return

    end_ts       = start_ts + pd.Timedelta(days=_BT_WINDOW - 1)
    buffer_start = start_ts - pd.Timedelta(days=100)  # ~70 trading-day lookback

    # ── Smart fetch: only download if sim_db doesn't cover the needed window ─
    _min_dt, _max_dt = sim_db.get_prices_date_range()
    _needs_fetch = (
        _min_dt is None                                     # no data at all
        or _max_dt < end_ts                                 # data ends before window
        or _min_dt > buffer_start + pd.Timedelta(days=15)  # not enough lookback
    )

    if _needs_fetch:
        fetch_start = (start_ts - pd.Timedelta(days=100)).strftime("%Y-%m-%d")
        fetch_end   = (end_ts   + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        st.markdown(_loading_overlay(
            "Setting Up Backtest",
            f"Fetching S&amp;P 500 data for {start_ts.strftime('%b %Y')}&hellip;",
        ), unsafe_allow_html=True)
        try:
            md.fetch_and_store_prices(sim_db, start=fetch_start, end=fetch_end)
            st.cache_data.clear()
        except Exception as exc:
            st.session_state["new_game_stage"]  = "bt_setup"
            st.session_state["_new_game_error"] = f"Price fetch failed: {exc}"
            st.rerun()
            return
    else:
        st.markdown(_loading_overlay("Setting Up Backtest", "Preparing historical data&hellip;"),
                    unsafe_allow_html=True)

    # ── Reset backtest DB ────────────────────────────────────────────────────
    try:
        db.reset_all(INITIAL_CAPITAL)
        st.cache_data.clear()
    except Exception as exc:
        st.session_state["new_game_stage"]  = "bt_setup"
        st.session_state["_new_game_error"] = str(exc)
        st.rerun()
        return

    # ── Copy relevant price window into backtest DB ──────────────────────────
    try:
        prices_window = sim_db.load_prices(cutoff_date=end_ts)
        if not prices_window.empty:
            prices_window = prices_window[prices_window.index >= buffer_start]
            db.upsert_prices(prices_window)

        # Take exactly SIMULATION_DAYS trading days starting from start_ts
        bt_dates = sorted([d for d in prices_window.index if d >= start_ts])[:SIMULATION_DAYS]
    except Exception as exc:
        st.session_state["new_game_stage"]  = "bt_setup"
        st.session_state["_new_game_error"] = f"Price data load failed: {exc}"
        st.rerun()
        return

    if not bt_dates:
        st.session_state["new_game_stage"]  = "bt_setup"
        st.session_state["_new_game_error"] = "No trading dates found in the selected range."
        st.rerun()
        return

    st.session_state.update({
        "new_game_stage":   "bt_ready",   # show welcome screen before first trade
        "app_mode":         "backtest",
        "sim_started":      True,
        "sim_date_idx":     0,
        "sim_day_num":      1,
        "sim_dates":        bt_dates,
        "daily_results":    [],
        "sim_capital":      INITIAL_CAPITAL,
        "sim_strategy":     "Momentum",
        "report_date":      "",
        "portfolio_report": None,
        "pending_day_data": None,
        "agent_logs":       [],
        "auto_deal":        True,
    })
    st.rerun()


def _render_bt_ready() -> None:
    """
    Full-page welcome screen shown after backtest setup completes —
    before the first trade is executed.  The user clicks 'Start Backtesting'
    to advance to the live backtest view.
    """
    bt_start_str = st.session_state.get("bt_start_date", "")
    bt_dates     = st.session_state.get("sim_dates", [])
    n_days       = len(bt_dates)

    try:
        _start_ts = pd.Timestamp(bt_start_str)
        _end_ts   = pd.Timestamp(bt_dates[-1]) if bt_dates else _start_ts
        date_range_label = (
            f"{_start_ts.strftime('%d %b %Y')} → {_end_ts.strftime('%d %b %Y')}"
        )
    except Exception:
        date_range_label = bt_start_str

    st.markdown("""
<style>
header,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{padding:0!important;max-width:780px!important;margin:0 auto!important;}
.stButton button {
    background:linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%)!important;
    color:#fff!important;font-weight:900!important;font-size:1.05rem!important;
    text-transform:uppercase!important;letter-spacing:0.22em!important;
    padding:0.9rem 1rem!important;border-radius:10px!important;border:none!important;
    box-shadow:0 0 32px rgba(59,130,246,0.4),0 0 64px rgba(59,130,246,0.12)!important;
    transition:box-shadow 0.2s,transform 0.15s!important;
}
.stButton button:hover {
    box-shadow:0 0 55px rgba(59,130,246,0.65),0 0 110px rgba(59,130,246,0.28)!important;
    transform:translateY(-2px)!important;
}
@keyframes bt-glow{0%,100%{filter:drop-shadow(0 0 18px rgba(59,130,246,0.4));}
                   50%{filter:drop-shadow(0 0 36px rgba(59,130,246,0.8));}}
@keyframes bt-fade{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
</style>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="padding:4vh 1.5rem 1.5rem;text-align:center;animation:bt-fade 0.55s ease-out;">

  <div style="font-size:5rem;animation:bt-glow 2.5s ease-in-out infinite;
              display:inline-block;margin-bottom:1rem;">⏱️</div>

  <div style="font-size:0.68rem;font-weight:800;text-transform:uppercase;
              letter-spacing:0.4em;color:#60a5fa;margin-bottom:0.6rem;">Backtest Ready</div>

  <h1 style="font-size:2.8rem;font-weight:900;color:#f1f5f9;text-transform:uppercase;
             letter-spacing:0.05em;margin:0 0 0.4rem;line-height:1.1;">
    Time Machine Active
  </h1>

  <p style="color:#9ca3af;font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:500;margin:0 0 1.8rem;">AI-Powered Historical Replay</p>

  <div style="display:inline-flex;gap:2rem;justify-content:center;
              background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.18);
              border-radius:14px;padding:1rem 2.5rem;margin-bottom:2rem;">
    <div style="text-align:center;">
      <div style="font-size:0.55rem;font-weight:800;color:#60a5fa;text-transform:uppercase;
                  letter-spacing:0.15em;margin-bottom:0.3rem;">Window</div>
      <div style="font-size:1rem;font-weight:900;font-family:monospace;color:#f1f5f9;">
        {date_range_label}
      </div>
    </div>
    <div style="width:1px;background:rgba(59,130,246,0.2);"></div>
    <div style="text-align:center;">
      <div style="font-size:0.55rem;font-weight:800;color:#60a5fa;text-transform:uppercase;
                  letter-spacing:0.15em;margin-bottom:0.3rem;">Trading Days</div>
      <div style="font-size:2rem;font-weight:900;font-family:monospace;color:#f1f5f9;">
        {n_days}
      </div>
    </div>
    <div style="width:1px;background:rgba(59,130,246,0.2);"></div>
    <div style="text-align:center;">
      <div style="font-size:0.55rem;font-weight:800;color:#60a5fa;text-transform:uppercase;
                  letter-spacing:0.15em;margin-bottom:0.3rem;">Starting Capital</div>
      <div style="font-size:1rem;font-weight:900;font-family:monospace;color:#f1f5f9;">
        ${INITIAL_CAPITAL:,.0f}
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button("⏱️  Start Backtesting", type="primary", use_container_width=True):
            st.session_state["new_game_stage"] = None
            st.rerun()

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    _, back_col, _ = st.columns([1, 2, 1])
    with back_col:
        if st.button("← Change Date", use_container_width=True):
            st.session_state["new_game_stage"] = "bt_setup"
            st.rerun()


def _render_new_game_hero() -> None:
    """Full-page hero/onboarding screen — shown when user clicks Start New Game."""
    st.markdown("""
<style>
header,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{padding:0!important;max-width:780px!important;margin:0 auto!important;}
.stButton button {
    background:linear-gradient(135deg,#f59e0b 0%,#d97706 100%)!important;
    color:#0a0a0a!important;font-weight:900!important;font-size:1.05rem!important;
    text-transform:uppercase!important;letter-spacing:0.22em!important;
    padding:0.9rem 1rem!important;border-radius:10px!important;border:none!important;
    box-shadow:0 0 32px rgba(245,158,11,0.4),0 0 64px rgba(245,158,11,0.12)!important;
    transition:box-shadow 0.2s,transform 0.15s!important;
}
.stButton button:hover {
    box-shadow:0 0 55px rgba(245,158,11,0.65),0 0 110px rgba(245,158,11,0.28)!important;
    transform:translateY(-2px)!important;
}
@keyframes ng-glow{0%,100%{filter:drop-shadow(0 0 18px rgba(245,158,11,0.4));}
                   50%{filter:drop-shadow(0 0 36px rgba(245,158,11,0.8));}}
@keyframes ng-fade{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
</style>
""", unsafe_allow_html=True)

    _err = st.session_state.pop("_new_game_error", None)
    if _err:
        st.error(f"Market data fetch failed: {_err}  —  please try again.")

    st.markdown(f"""
<div style="padding:3vh 1.5rem 1.5rem;text-align:center;animation:ng-fade 0.55s ease-out;">

  <div style="font-size:5.5rem;animation:ng-glow 2.5s ease-in-out infinite;
              display:inline-block;margin-bottom:1.6rem;">🎰</div>

  <div style="font-size:0.68rem;font-weight:800;text-transform:uppercase;
              letter-spacing:0.4em;color:#f59e0b;margin-bottom:0.7rem;">Welcome to</div>

  <h1 style="font-size:3.1rem;font-weight:900;color:#f1f5f9;text-transform:uppercase;
             letter-spacing:0.05em;margin:0 0 0.45rem;line-height:1.1;">
    The Trading Floor
  </h1>

  <p style="color:#9ca3af;font-size:0.9rem;letter-spacing:0.12em;text-transform:uppercase;
            font-weight:500;margin:0 0 2.8rem;">AI-Powered Paper Trading Simulator</p>

  <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:3rem;">
    <div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.22);
                border-radius:12px;padding:1rem 1.5rem;min-width:125px;">
      <div style="font-size:1.45rem;font-weight:900;font-family:monospace;
                  color:#f59e0b;line-height:1;">${INITIAL_CAPITAL:,.0f}</div>
      <div style="font-size:0.56rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.14em;color:#6b7280;margin-top:5px;">Starting Capital</div>
    </div>
    <div style="background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.22);
                border-radius:12px;padding:1rem 1.5rem;min-width:125px;">
      <div style="font-size:1.45rem;font-weight:900;font-family:monospace;
                  color:#10b981;line-height:1;">500+</div>
      <div style="font-size:0.56rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.14em;color:#6b7280;margin-top:5px;">S&amp;P 500 Universe</div>
    </div>
    <div style="background:rgba(139,92,246,0.07);border:1px solid rgba(139,92,246,0.22);
                border-radius:12px;padding:1rem 1.5rem;min-width:125px;">
      <div style="font-size:1.45rem;font-weight:900;font-family:monospace;
                  color:#8b5cf6;line-height:1;">Claude</div>
      <div style="font-size:0.56rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.14em;color:#6b7280;margin-top:5px;">AI Strategy</div>
    </div>
    <div style="background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.22);
                border-radius:12px;padding:1rem 1.5rem;min-width:125px;">
      <div style="font-size:1.45rem;font-weight:900;font-family:monospace;
                  color:#3b82f6;line-height:1;">3–5</div>
      <div style="font-size:0.56rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.14em;color:#6b7280;margin-top:5px;">Max Positions</div>
    </div>
  </div>

  <p style="color:#4b5563;font-size:0.65rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.18em;line-height:1.8;margin:0 0 1.6rem;">
    High-Volatility Mean Reversion&nbsp;·&nbsp;S&amp;P 500 Universe<br>
    Equal-Weight Daily Rebalance&nbsp;·&nbsp;Risk-Managed by AI
  </p>

</div>
""", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        if st.button("🎲  Begin Your Journey", type="primary", use_container_width=True):
            st.session_state["new_game_stage"] = "fetching"
            st.rerun()


_LOADING_SCREEN_CSS = """
<style>
/* Hide all Streamlit chrome so nothing bleeds through loading screens */
header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stSidebar"],#MainMenu,.viewerBadge_container__r5tak
{ display:none !important; }
/* Make every container behind our content transparent / zero-padded */
[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],section.main,.main
{ background:#0a0a0a !important; padding:0 !important; }
.block-container,[data-testid="stMainBlockContainer"]
{ padding:0 !important; max-width:100% !important; background:#0a0a0a !important; }
@keyframes ld-spin{to{transform:rotate(360deg);}}
@keyframes ld-bounce{0%,80%,100%{transform:scale(0.4);opacity:0.25;}
                     40%{transform:scale(1);opacity:1;}}
.ld-dot{display:inline-block;width:10px;height:10px;border-radius:50%;
        background:#f59e0b;animation:ld-bounce 1.3s infinite ease-in-out;}
</style>
"""

def _loading_overlay(title: str, subtitle: str) -> str:
    """
    Full-screen loading card.  Uses min-height:100vh in normal document flow
    (NOT position:fixed) so it works correctly inside Streamlit's CSS-contained
    block structure without overlapping or leaking other page elements.
    """
    return f"""{_LOADING_SCREEN_CSS}
<div style="min-height:100vh;width:100%;background:#0a0a0a;
            display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:1.6rem;">
  <div style="font-size:5.5rem;animation:ld-spin 1.4s linear infinite;
              display:inline-block;filter:drop-shadow(0 0 20px rgba(245,158,11,0.4));">🎰</div>
  <div style="text-align:center;">
    <h1 style="font-weight:900;color:#f1f5f9;text-transform:uppercase;
               letter-spacing:0.08em;font-size:1.85rem;margin:0 0 0.5rem;">{title}</h1>
    <div style="color:#f59e0b;font-size:0.75rem;font-weight:800;
                text-transform:uppercase;letter-spacing:0.28em;">{subtitle}</div>
  </div>
  <div style="display:flex;gap:7px;margin-top:0.2rem;">
    <span class="ld-dot"></span>
    <span class="ld-dot" style="animation-delay:.18s"></span>
    <span class="ld-dot" style="animation-delay:.36s"></span>
  </div>
</div>"""


def _reset_game_with_loading(db: "PortfolioDatabase") -> None:
    """
    Immediately reset the DB and all session state, then show the hero screen.
    Runs as soon as 'Start New Game' is clicked — no market fetch yet.

    Uses db.reset() (keeps price history) so the subsequent _start_new_game_with_loading
    can skip the full re-download and only top-up missing days.
    """
    st.markdown(_loading_overlay("Clearing the Table", "Resetting your portfolio&hellip;"),
                unsafe_allow_html=True)

    try:
        db.reset(INITIAL_CAPITAL)   # keep prices — game state only
        st.cache_data.clear()
    except Exception as exc:
        st.session_state["new_game_stage"]  = "mode_select"
        st.session_state["_new_game_error"] = str(exc)
        st.rerun()
        return

    st.session_state.update({
        "new_game_stage":   "fetching",
        "app_mode":         "simulation",
        "sim_started":      False,
        "sim_date_idx":     0,
        "sim_day_num":      1,
        "sim_dates":        [],
        "daily_results":    [],
        "sim_capital":      INITIAL_CAPITAL,
        "sim_strategy":     "Momentum",
        "report_date":      "",
        "portfolio_report": None,
        "pending_day_data": None,
        "agent_logs":       [],
    })
    st.rerun()


def _start_new_game_with_loading(db: "PortfolioDatabase") -> None:
    """
    Show a loading screen while syncing market data.
    DB is already reset (game state only) by _reset_game_with_loading.
    Reuses any prices already in portfolio.db or backtest.db — only
    fetches genuinely missing data from yfinance.
    """
    st.markdown(_loading_overlay(
        "Setting Up The Floor",
        "Syncing latest market data&hellip;",
    ), unsafe_allow_html=True)

    try:
        fresh_df = _sync_sim_prices(db)   # smart: copy/top-up, no full re-download
        st.cache_data.clear()
    except Exception as exc:
        st.session_state["new_game_stage"]  = "mode_select"
        st.session_state["_new_game_error"] = str(exc) or "Failed to sync market data."
        st.rerun()
        return

    fresh_df.index = pd.to_datetime(fresh_df.index)
    fresh_df = fresh_df.sort_index()
    sim_dates = fresh_df.index[-SIMULATION_DAYS:].tolist()

    st.session_state.update({
        "new_game_stage":   None,
        "app_mode":         "simulation",
        "sim_started":      True,
        "sim_date_idx":     0,
        "sim_day_num":      1,
        "sim_dates":        sim_dates,
        "daily_results":    [],
        "sim_capital":      INITIAL_CAPITAL,
        "sim_strategy":     "Momentum",
        "report_date":      "",
        "portfolio_report": None,
        "pending_day_data": None,
        "agent_logs":       [],
        "auto_deal":        True,
    })
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Reset-guard screen
# ═══════════════════════════════════════════════════════════════════════════════

def _confirm_reset_screen() -> None:
    """Full-page password gate — shown instead of the modal dialog for reliability."""
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            """
            <div style="text-align:center; padding:3rem 0 1.5rem;">
              <div style="font-size:3rem; margin-bottom:0.75rem;">🔐</div>
              <div style="font-size:1.25rem; font-weight:900; color:#f1f5f9;
                          text-transform:uppercase; letter-spacing:0.06em;
                          margin-bottom:0.5rem;">Confirm Reset</div>
              <div style="color:#9ca3af; font-size:0.85rem;
                          margin-bottom:1.25rem;">
                Enter the secret word to wipe all portfolio data and start fresh.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pwd = st.text_input(
            "Secret word",
            type="password",
            placeholder="Enter secret word…",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅  Confirm", type="primary", use_container_width=True):
                if pwd == RESET_PASSWORD:
                    st.session_state["_reset_confirmed"] = True
                    st.rerun()
                else:
                    st.error("❌  Incorrect — try again.")
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["new_game_stage"] = "mode_select"
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Loading screen + price history fetch on first browser visit ───────────
    # The loading screen HTML renders immediately while _ensure_price_history()
    # runs in the background. After the fetch completes, st.rerun() takes the
    # user straight to the mode-select screen with prices ready.
    if "app_initialized" not in st.session_state:
        st.session_state["app_initialized"] = True
        _show_loading_screen()
        _ensure_price_history()   # fetch/refresh while the splash is visible
        st.rerun()
        return

    # ── Choose DB based on mode (defaults to simulation) ─────────────────────
    app_mode = st.session_state.get("app_mode", "simulation")
    db = PortfolioDatabase(BACKTEST_DB_PATH if app_mode == "backtest" else DB_PATH)

    # ── Restore simulation state from DB on fresh page load ──────────────────
    # Must run BEFORE the defaults loop so restored keys are not overwritten.
    _restore_session_from_db(db)

    # ── New-game onboarding screens (intercept before anything else) ──────────
    # On every fresh browser session (page reload), the key won't exist yet —
    # always land on mode_select so the user can consciously choose.
    # Active-session detection is surfaced as "Continue" buttons inside that screen.
    if "new_game_stage" not in st.session_state:
        st.session_state["new_game_stage"] = "mode_select"

    _new_game_stage = st.session_state.get("new_game_stage")
    if _new_game_stage == "mode_select":
        _render_mode_select()
        return
    if _new_game_stage == "resetting":
        # Password gate — show full-page screen until confirmed, then proceed
        if not st.session_state.pop("_reset_confirmed", False):
            _confirm_reset_screen()
            return
        _reset_game_with_loading(db)
        return
    if _new_game_stage == "fetching":
        _start_new_game_with_loading(db)
        return
    if _new_game_stage == "bt_setup":
        _render_bt_setup()
        return
    if _new_game_stage == "bt_resetting":
        _start_backtest_with_loading(db)
        return
    if _new_game_stage == "bt_ready":
        _render_bt_ready()
        return

    # Load price data from the active DB's prices table
    _prices_ready = db.has_prices()
    prices_df  = load_price_data(db.db_path) if _prices_ready else None

    # ── Session state init (only sets keys not yet present) ──────────────────
    defaults = {
        "app_mode":          "simulation",  # "simulation" | "backtest"
        "bt_start_date":     "",            # ISO date string for backtest start
        "sim_started":       False,
        "sim_date_idx":      0,
        "sim_day_num":       1,
        "sim_dates":         [],
        "daily_results":     [],
        "sim_capital":       INITIAL_CAPITAL,
        "sim_strategy":      "Momentum",
        "portfolio_report":  None,
        "report_date":       "",
        "pending_day_data":  None,
        "new_game_stage":    None,
        # roulette
        "roulette_wins":     0,
        "roulette_losses":   0,
        "roulette_streak":   0,
        "roulette_result":   None,
        # agent pipeline logs
        "agent_logs":        [],
    }
    # Keep app_mode in sync after the defaults loop sets it
    app_mode = st.session_state.get("app_mode", "simulation")
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    today_str = date.today().isoformat()

    # ── Show trade confirmation screen if a proposal is pending ─────────────
    # Renders inline (full-page) — returns early so nothing else renders.
    if st.session_state.get("pending_day_data") is not None:
        show_proposals_dialog(db, today_str)
        return

    # ── Auto-generate portfolio report (once per day in sim; once per step in bt)
    # Report key changes every backtest step so we re-run after each advance.
    if app_mode == "backtest":
        _report_key = f"bt_day_{st.session_state.get('sim_day_num', 0)}"
    else:
        _report_key = today_str

    if _prices_ready and st.session_state["report_date"] != _report_key:
        with st.spinner("Generating daily briefing …"):
            try:
                data_agent = DataAgent()
                # In backtest mode, only show prices up to the last completed step
                _prices_for_report = prices_df
                if app_mode == "backtest" and prices_df is not None:
                    _bt_dates_list = st.session_state.get("sim_dates", [])
                    _bt_idx        = st.session_state.get("sim_date_idx", 0)
                    if 0 < _bt_idx <= len(_bt_dates_list):
                        _last_bt = pd.Timestamp(_bt_dates_list[_bt_idx - 1])
                        _prices_for_report = prices_df[prices_df.index <= _last_bt]
                st.session_state["portfolio_report"] = data_agent.generate_portfolio_report(
                    db, _prices_for_report
                )
                st.session_state["report_date"] = _report_key
            except Exception as exc:
                st.session_state["portfolio_report"] = None
                logger.warning("Portfolio report failed: %s", exc)

    # ── Compute advance flags (needed for control strip + auto-deal) ─────────
    if app_mode == "backtest":
        # No time gates in backtest — advance freely through historical dates
        already_ran_today = False
        _is_weekend       = False
        dates_remaining   = (
            st.session_state["sim_started"]
            and st.session_state["sim_date_idx"] < len(st.session_state.get("sim_dates", []))
        )
    else:
        already_ran_today = db.get_last_advance_date() == today_str
        _is_weekend       = date.today().weekday() >= 5   # 5=Sat, 6=Sun
        dates_remaining   = (
            st.session_state["sim_started"]
            and st.session_state["sim_date_idx"] < SIMULATION_DAYS
        )

    pending_confirm = st.session_state.get("pending_day_data") is not None
    # Day-advancement gate: weekends block advancing to the next trading day,
    # but the initial auto-deal on simulation start is allowed any day.
    can_advance = (
        dates_remaining
        and not already_ran_today
        and not pending_confirm
        and not _is_weekend
        and prices_df is not None
    )
    # Auto-deal gate: same as can_advance but without the weekend restriction
    # so a new simulation can be started and the first hand dealt on any day.
    _can_auto_deal = (
        dates_remaining
        and not already_ran_today
        and not pending_confirm
        and prices_df is not None
    )

    def _do_deal():
        """Call _deal_next_hand with the right date (backtest or live)."""
        if app_mode == "backtest":
            _idx  = st.session_state["sim_date_idx"]
            _date = st.session_state["sim_dates"][_idx]
            _deal_next_hand(db, backtest_date=_date)
        else:
            _deal_next_hand(db)

    # Auto-deal: fires once on the rerun immediately after Start New Game.
    # Uses _can_auto_deal (no weekend restriction) so starting on Sat/Sun works.
    if st.session_state.pop("auto_deal", False) and _can_auto_deal:
        _do_deal()

    # ─────────────────────────────────────────────────────────────────────────
    # ── Title
    # ─────────────────────────────────────────────────────────────────────────
    _mode_span = (
        " · <span style='color:#60a5fa;'>Back Testing Mode</span>"
        if app_mode == "backtest"
        else " · <span style='color:#f59e0b;'>Paper Trading Simulator</span>"
    )
    _left_pad, _title_col, _casino_col, _right_pad = st.columns([2, 14, 1, 2])
    with _title_col:
        st.markdown(
            f"<h1 style='font-size:2.1rem; font-weight:900; color:#f1f5f9; "
            f"text-transform:uppercase; letter-spacing:0.04em; padding-top:1rem; "
            f"margin-bottom:0; text-align:center;'>"
            f"AI Investment Application{_mode_span}"
            f"</h1>",
            unsafe_allow_html=True,
        )
    with _casino_col:
        st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
        with st.popover("🎰", use_container_width=False):
            render_roulette()

    # ─────────────────────────────────────────────────────────────────────────
    # ── Persistent control strip (always visible, cannot be hidden)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="ctrl-top-border"></div>', unsafe_allow_html=True)

    # ── Date / day label logic ────────────────────────────────────────────────
    if app_mode == "backtest":
        _bt_dates_list = st.session_state.get("sim_dates", [])
        _bt_idx        = st.session_state.get("sim_date_idx", 0)
        _card_label    = "📅 Backtest Date"
        if _bt_idx < len(_bt_dates_list):
            _bt_next   = pd.Timestamp(_bt_dates_list[_bt_idx])
            _date_part = _bt_next.strftime("%d %B %Y").lstrip("0")
        else:
            _date_part = "Complete"
        _day_label = (
            f"BACKTEST · DAY {st.session_state['sim_day_num'] - 1}"
            if st.session_state.get("daily_results") else "BACKTEST · DAY —"
        )
        _card_border = "rgba(96,165,250,0.22)"
        _card_label_color = "#60a5fa"
    else:
        _today      = date.today()
        _date_part  = f"{_ordinal(_today.day)} {_today.strftime('%B %Y')}"
        _card_label = "📅 Trading Date"
        _day_label  = (
            f"DAY {st.session_state['sim_day_num'] - 1}"
            if st.session_state["sim_started"] and st.session_state["daily_results"]
            else "DAY —"
        )
        _card_border      = "rgba(245,158,11,0.22)"
        _card_label_color = "#f59e0b"

    # ── Row 1: Full-width info cards ──────────────────────────────────────────
    c_date, c_deal = st.columns(2)

    with c_date:
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.025); border:1px solid {_card_border}; "
            f"border-radius:12px; padding:0.85rem 1.2rem; text-align:center; "
            f"min-height:110px; display:flex; flex-direction:column; justify-content:center;'>"
            f"<div style='font-size:0.58rem; font-weight:800; color:{_card_label_color}; "
            f"text-transform:uppercase; letter-spacing:0.18em; margin-bottom:0.45rem;'>{_card_label}</div>"
            f"<div style='font-family:monospace; font-size:1.1rem; font-weight:900; color:#f1f5f9; "
            f"letter-spacing:0.02em;'>{_date_part}</div>"
            f"<div style='font-size:0.65rem; font-weight:700; color:#6b7280; margin-top:0.2rem; "
            f"text-transform:uppercase; letter-spacing:0.12em;'>{_day_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c_deal:
        _next_move_color = "#60a5fa" if app_mode == "backtest" else "#f59e0b"
        _next_move_label = "⏱️ Next Step" if app_mode == "backtest" else "📈 Next Move"
        # CSS: force the bordered container to the same min-height as the left card.
        # The marker div and stVerticalBlockBorderWrapper are SIBLINGS inside the
        # column's stVerticalBlock, so we use :has() to scope the selector correctly.
        st.markdown(
            "<style>"
            "[data-testid='stVerticalBlock']:has(.ctrl-right-card) "
            "[data-testid='stVerticalBlockBorderWrapper']"
            "{ min-height:110px !important; }"
            "[data-testid='stVerticalBlock']:has(.ctrl-right-card) "
            "[data-testid='stVerticalBlockBorderWrapper'] > div"
            "{ min-height:110px !important; display:flex !important; "
            "flex-direction:column !important; justify-content:center !important; }"
            "</style>"
            "<div class='ctrl-right-card'></div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:0.58rem; font-weight:800; color:{_next_move_color}; "
                "text-transform:uppercase; letter-spacing:0.18em; margin-bottom:0.35rem; "
                f"text-align:center;'>{_next_move_label}</div>",
                unsafe_allow_html=True,
            )
            if st.session_state["sim_started"] and not dates_remaining:
                st.success("All hands dealt", icon="🏁")
            else:
                if app_mode != "backtest" and _is_weekend and dates_remaining:
                    st.markdown(
                        "<div style='font-size:0.65rem; color:#f59e0b; font-weight:700; "
                        "text-align:center; padding-bottom:0.2rem;'>🗓️ Markets closed — try Monday</div>",
                        unsafe_allow_html=True,
                    )
                elif app_mode != "backtest" and already_ran_today and dates_remaining:
                    st.markdown(
                        "<div style='font-size:0.65rem; color:#f59e0b; font-weight:700; "
                        "text-align:center; padding-bottom:0.2rem;'>⏳ Come back tomorrow</div>",
                        unsafe_allow_html=True,
                    )
                _btn_label = "⏱️  Next Backtest Day" if app_mode == "backtest" else "📈  Deal Next Hand"
                if st.button(_btn_label, disabled=not can_advance, use_container_width=True):
                    _do_deal()

    st.markdown('<div class="ctrl-bottom-border"></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ── Main tabs
    # ─────────────────────────────────────────────────────────────────────────
    tab_summary, tab_portfolio, tab_market, tab_logs = st.tabs([
        "🗓️  Daily Summary",
        "📊  Portfolio",
        "🌐  Market Data",
        "🤖  Agent Logs",
    ])

    daily_results    = st.session_state["daily_results"]
    strategy_lbl     = st.session_state["sim_strategy"]
    init_cap         = st.session_state["sim_capital"]
    portfolio_report = st.session_state.get("portfolio_report")

    with tab_summary:
        render_daily_summary(daily_results, strategy_lbl, portfolio_report)

    with tab_portfolio:
        render_portfolio(db, daily_results, init_cap, portfolio_report)

    with tab_market:
        if app_mode == "backtest":
            # backtest DB prices table is already scoped to this backtest window
            _bt_prices     = load_price_data(db.db_path)
            _bt_dates_done = st.session_state.get("sim_dates", [])
            _bt_idx        = st.session_state.get("sim_date_idx", 0)
            _bt_cutoff     = _bt_dates_done[_bt_idx - 1] if _bt_idx > 0 and _bt_dates_done else None
            render_market_data(db, bt_prices_df=_bt_prices, bt_cutoff_date=_bt_cutoff)
        else:
            render_market_data(db)

    with tab_logs:
        render_agent_logs()

    # ─────────────────────────────────────────────────────────────────────────
    # ── Footer (always at the very bottom of the page)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="ctrl-top-border" style="margin-top:2rem;"></div>',
                unsafe_allow_html=True)

    f_start, f_api = st.columns(2)

    with f_start:
        if st.button("🎲  Start New Game", type="primary", use_container_width=True):
            # Already in simulation — reset and go straight to fetching.
            # To switch modes, refresh the page instead.
            st.session_state["app_mode"]       = "simulation"
            st.session_state["new_game_stage"] = "resetting"
            st.rerun()

    with f_api:
        if API_KEY:
            _api_html = (
                "<div style='background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.35); "
                "border-radius:6px; padding:0.45rem 1rem; text-align:center; "
                "font-weight:600; color:#10b981; font-size:0.9rem; width:100%;'>"
                "🤖&nbsp;&nbsp;Claude AI · connected</div>"
            )
        else:
            _api_html = (
                "<div style='background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.35); "
                "border-radius:6px; padding:0.45rem 1rem; text-align:center; "
                "font-weight:600; color:#f59e0b; font-size:0.9rem; width:100%;'>"
                "⚙️&nbsp;&nbsp;Rule-based mode</div>"
            )
        st.markdown(_api_html, unsafe_allow_html=True)

    st.markdown('<div class="ctrl-bottom-border"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

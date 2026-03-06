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
    padding: 0.2rem 0.4rem !important;
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

    return (
        f"<div style='padding:0.55rem 0.1rem;'>"
        f"<div style='font-size:0.65rem; font-weight:800; text-transform:uppercase; "
        f"letter-spacing:0.14em; color:#f59e0b; margin-bottom:0.25rem;'>{label}</div>"
        f"<div style='display:flex; align-items:baseline; flex-wrap:wrap; gap:0;'>"
        f"<span style='font-family:\"Courier New\",monospace; font-size:1.65rem; "
        f"font-weight:900; color:#f1f5f9; line-height:1;'>{value}</span>"
        f"{delta_html}"
        f"</div>"
        f"</div>"
    )


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
# Trade Proposals Dialog
# ═══════════════════════════════════════════════════════════════════════════════

@st.dialog("📋  Proposed Trades — Confirm to Execute", width="large")
def show_proposals_dialog(db: "PortfolioDatabase", today_str: str) -> None:
    """
    Modal dialog that shows the proposed trades for the current simulation day
    and lets the user confirm or cancel execution.
    """
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
        st.markdown("#### 🟢  New positions (equal-weight, 20% each)")
        buy_rows = []
        for p in buys:
            m     = analysis.get(p["ticker"], {})
            price = m.get("price", 0.0)
            vol   = m.get("realized_vol_5d", 0.0)
            chg   = m.get("prev_day_return", 0.0)
            amt   = total_val * 0.20
            buy_rows.append({
                "Ticker":             p["ticker"],
                "Price":              f"${price:,.2f}",
                "5d Realised Vol":    f"{vol:.1%}",
                "Prev Day Change":    f"{chg:+.2f}%",
                "Amount to Invest":   f"${amt:,.0f}",
            })
        st.dataframe(
            buy_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Strategy: Top-50 S&P 500 stocks by 5-day realised volatility, "
            "then the 5 with the largest previous-day loss."
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
            prices_df = load_price_data()

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

            db.save_day_result(result)
            db.set_last_advance_date(today_str)
            st.session_state["daily_results"].append(result)
            st.session_state["sim_date_idx"] += 1
            st.session_state["sim_day_num"]  += 1
            st.session_state["report_date"]   = ""   # force report refresh
            st.session_state["pending_day_data"] = None
            st.rerun()

    with col_cancel:
        if st.button("❌  Cancel", use_container_width=True):
            st.session_state["pending_day_data"] = None
            st.rerun()


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
    if not spy_raw.empty:
        spy_a = spy_raw.reindex(port_ext.index, method="ffill").dropna()
        if len(spy_a) >= 1:
            spy_norm = spy_a / spy_a.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=spy_norm.index, y=spy_norm.values,
                name="S&P 500 (SPY)", mode="lines+markers",
                line=dict(color="#60a5fa", width=2, dash="dot"),
                marker=dict(size=5, color="#60a5fa"),
            ))
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(107,114,128,0.35)")
    fig.update_layout(
        template=_DARK, height=190,
        margin=dict(l=0, r=0, t=6, b=0),
        yaxis=dict(title="Indexed (100 = start)", tickformat=".2f"),
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

        if not pd.isna(pv) and not pd.isna(sv):
            win   = pv > sv if higher_better else pv < sv
            color = "#10b981" if win else "#ef4444"
            arrow = "▲" if win else "▼"
        else:
            color = "#f59e0b" if not pd.isna(pv) else "#6b7280"
            arrow = ""

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

    day      = daily_results[-1]
    snap     = day["portfolio"]
    init_cap = st.session_state["sim_capital"]

    # ── Header: 4 KPI metrics ─────────────────────────────────────────────────
    prev_val      = daily_results[-2]["portfolio"]["total_value"] if len(daily_results) > 1 else init_cap
    today_pnl     = snap["total_value"] - prev_val
    today_pnl_pct = (today_pnl / prev_val * 100) if prev_val > 0 else 0.0

    st.markdown(
        f"<div style='font-size:0.65rem; font-weight:800; text-transform:uppercase; "
        f"letter-spacing:0.14em; color:#f59e0b; margin-bottom:0.6rem;'>"
        f"📅 Day {day['day_num']}  ·  {day['date']}</div>",
        unsafe_allow_html=True,
    )

    # ── Header KPI row (inline deltas via custom HTML) ────────────────────────
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
        _pos = True if today_pnl > 0 else (False if today_pnl < 0 else None)
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
                st.markdown(portfolio_report["summary"])
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
            _pos2 = True if r['fiveday_pnl'] > 0 else (False if r['fiveday_pnl'] < 0 else None)
            st.markdown(
                _html_metric("5-Day P/L", f"{r['fiveday_pnl_pct']:+.2f}%",
                             delta=f"${r['fiveday_pnl']:+,.0f}", delta_positive=_pos2),
                unsafe_allow_html=True,
            )
        with h3:
            _pos3 = True if r['alltime_pnl'] > 0 else (False if r['alltime_pnl'] < 0 else None)
            st.markdown(
                _html_metric("All-Time P/L", f"{r['alltime_pnl_pct']:+.2f}%",
                             delta=f"${r['alltime_pnl']:+,.0f}", delta_positive=_pos3),
                unsafe_allow_html=True,
            )
        with h4:
            st.markdown(_html_metric("Win Rate", f"{r['win_rate']:.0f}%"),
                        unsafe_allow_html=True)
    else:
        st.markdown(_html_metric("Portfolio Value", f"${snap['total_value']:,.0f}"),
                    unsafe_allow_html=True)

    st.divider()

    # ── USD Portfolio Value graph (equity curve) ──────────────────────────────
    history_df = db.get_portfolio_history()
    if not history_df.empty:
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


def render_market_data(db: PortfolioDatabase):
    st.markdown(
        "<div style='font-size:0.72rem; font-weight:800; text-transform:uppercase; "
        "letter-spacing:0.12em; color:#f59e0b; margin-bottom:0.75rem;'>"
        "S&P 500 Live Market Data</div>",
        unsafe_allow_html=True,
    )
    st.caption("Fetches all S&P 500 constituents with latest OHLCV via yfinance. Cached in local DB.")

    fetch_dates = db.get_sp500_fetch_dates()

    if not fetch_dates:
        st.info("No data yet — press **Fetch / Refresh** below to download today's quotes.")
        if st.button("🔄  Fetch / Refresh", type="primary", use_container_width=True):
            with st.spinner("Pulling quotes from Yahoo Finance …"):
                try:
                    md.fetch_and_store_sp500(db)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")
        return

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

    # ── Summary metrics (no delta badges) ─────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(_html_metric("Stocks",      str(total)),   unsafe_allow_html=True)
    with m2:
        st.markdown(_html_metric("Winners",     str(gainers)), unsafe_allow_html=True)
    with m3:
        st.markdown(_html_metric("Losers",      str(losers)),  unsafe_allow_html=True)
    with m4:
        st.markdown(_html_metric("Market Date", price_date),   unsafe_allow_html=True)

    if price_date != selected_date:
        st.caption(
            f"ℹ️  Prices quoted as of **{price_date}** (last trading day) — "
            f"fetched on {selected_date}."
        )

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
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

    st.divider()

    # ── Fetch / Refresh button at the bottom ──────────────────────────────────
    if st.button("🔄  Fetch / Refresh Market Data", type="primary", use_container_width=True):
        with st.spinner("Pulling quotes from Yahoo Finance …"):
            try:
                md.fetch_and_store_sp500(db)
                st.rerun()
            except Exception as exc:
                st.error(f"Failed: {exc}")


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

def _deal_next_hand(db: "PortfolioDatabase") -> None:
    """
    Run the full proposal pipeline for the next simulation day and store the
    result in st.session_state["pending_day_data"]. Finishes with st.rerun()
    so the trade-confirmation dialog fires on the next pass.
    """
    day_num      = st.session_state["sim_day_num"]
    strategy_now = st.session_state["sim_strategy"]

    # 1. Pull in the latest market close prices
    with st.spinner("Fetching latest market data …"):
        try:
            md.refresh_latest_day(CSV_PATH)
            st.cache_data.clear()
        except Exception as exc:
            st.warning(f"Could not refresh market data: {exc}")

    prices_df_fresh = load_price_data()
    sim_date        = prices_df_fresh.index.max()
    date_str        = sim_date.strftime("%Y-%m-%d")
    prices          = prices_df_fresh.loc[sim_date].to_dict()
    portfolio       = db.get_portfolio_state(prices)

    # 2. Run analysis + strategy (proposal phase only — execution waits for
    #    user confirmation in the dialog)
    _spinner_msg = (
        f"Asking Claude to propose Day {day_num} trades …"
        if API_KEY
        else f"Analysing Day {day_num} (deterministic mode) …"
    )
    with st.spinner(_spinner_msg):
        data_agent     = DataAgent()
        strategy_agent = StrategyAgent(strategy_now)
        analysis       = data_agent.analyze(prices_df_fresh, sim_date)
        proposed       = strategy_agent.propose_trades(
            analysis, portfolio, date_str, day_num
        )

    # 3. Store payload; the dialog fires on the next rerun
    st.session_state["pending_day_data"] = {
        "proposals":  proposed,
        "analysis":   analysis,
        "sim_date":   sim_date,
        "prices":     prices,
        "strategy":   strategy_now,
        "reasoning":  strategy_agent.reasoning,
        "day_num":    day_num,
        "portfolio":  portfolio,
        "api_failed": strategy_agent.api_failed,
        "api_error":  strategy_agent.api_error,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    db = PortfolioDatabase()

    # ── Restore simulation state from DB on fresh page load ──────────────────
    # Must run BEFORE the defaults loop so restored keys are not overwritten.
    _restore_session_from_db(db)

    # Load price data only if the CSV exists (created by Start New Game)
    _csv_ready = Path(CSV_PATH).exists()
    prices_df  = load_price_data() if _csv_ready else None

    # ── Session state init (only sets keys not yet present) ──────────────────
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
        "pending_day_data":  None,   # set when waiting for trade confirmation
        # roulette
        "roulette_wins":     0,
        "roulette_losses":   0,
        "roulette_streak":   0,
        "roulette_result":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    today_str = date.today().isoformat()

    # ── Show trade confirmation dialog if a proposal is pending ──────────────
    if st.session_state.get("pending_day_data") is not None:
        show_proposals_dialog(db, today_str)

    # ── Auto-generate portfolio report once per calendar day ─────────────────
    # Only runs after Start New Game has populated the price CSV
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

    # ── Compute advance flags (needed for control strip + auto-deal) ─────────
    already_ran_today = db.get_last_advance_date() == today_str
    dates_remaining   = (
        st.session_state["sim_started"]
        and st.session_state["sim_date_idx"] < SIMULATION_DAYS
    )
    pending_confirm = st.session_state.get("pending_day_data") is not None
    can_advance = (
        dates_remaining
        and not already_ran_today
        and not pending_confirm
        and prices_df is not None
    )

    # Auto-deal: fires once on the rerun immediately after Start New Game
    if st.session_state.pop("auto_deal", False) and can_advance:
        _deal_next_hand(db)

    # ─────────────────────────────────────────────────────────────────────────
    # ── Title
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-size:2.1rem; font-weight:900; color:#f1f5f9; "
        "text-transform:uppercase; letter-spacing:0.04em; margin-top:0.4rem; "
        "margin-bottom:0; text-align:center;'>"
        "AI Investment Application"
        "<span style='color:#f59e0b;'> · Paper Trading Simulator</span></h1>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ── Persistent control strip (always visible, cannot be hidden)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="ctrl-top-border"></div>', unsafe_allow_html=True)

    _today     = date.today()
    _date_part = f"{_ordinal(_today.day)} {_today.strftime('%B %Y')}"
    if st.session_state["sim_started"] and st.session_state["daily_results"]:
        _day_label = f"DAY {st.session_state['sim_day_num'] - 1}"
    else:
        _day_label = "DAY —"

    # ── Row 1: Centered info cards ────────────────────────────────────────────
    _, c_date, c_deal, _ = st.columns([1, 2, 2, 1])

    with c_date:
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.03); border:1px solid rgba(245,158,11,0.28); "
            f"border-radius:12px; padding:0.9rem 1.2rem; text-align:center;'>"
            f"<div style='font-size:0.58rem; font-weight:800; color:#f59e0b; text-transform:uppercase; "
            f"letter-spacing:0.18em; margin-bottom:0.45rem;'>📅 Trading Date</div>"
            f"<div style='font-family:monospace; font-size:1.1rem; font-weight:900; color:#f1f5f9; "
            f"letter-spacing:0.02em;'>{_date_part}</div>"
            f"<div style='font-size:0.65rem; font-weight:700; color:#6b7280; margin-top:0.2rem; "
            f"text-transform:uppercase; letter-spacing:0.12em;'>{_day_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c_deal:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.58rem; font-weight:800; color:#f59e0b; "
                "text-transform:uppercase; letter-spacing:0.18em; margin-bottom:0.35rem; "
                "text-align:center;'>📈 Next Move</div>",
                unsafe_allow_html=True,
            )
            if st.session_state["sim_started"] and not dates_remaining:
                st.success("All hands dealt", icon="🏁")
            else:
                if already_ran_today and dates_remaining:
                    st.markdown(
                        "<div style='font-size:0.65rem; color:#f59e0b; font-weight:700; "
                        "text-align:center; padding-bottom:0.2rem;'>⏳ Come back tomorrow</div>",
                        unsafe_allow_html=True,
                    )
                if st.button("📈  Deal Next Hand", disabled=not can_advance,
                             use_container_width=True):
                    _deal_next_hand(db)

    # ── Casino section ────────────────────────────────────────────────────────
    st.markdown(
        "<div style='display:flex; align-items:center; gap:0.8rem; "
        "margin:1rem 0 0.6rem;'>"
        "<div style='flex:1; height:1px; background:linear-gradient(90deg, "
        "transparent, rgba(245,158,11,0.35));'></div>"
        "<div style='font-size:0.58rem; font-weight:800; color:#f59e0b; "
        "text-transform:uppercase; letter-spacing:0.22em; padding:0.2rem 0.75rem; "
        "border:1px solid rgba(245,158,11,0.3); border-radius:20px; "
        "background:rgba(245,158,11,0.06);'>🎰 Casino</div>"
        "<div style='flex:1; height:1px; background:linear-gradient(270deg, "
        "transparent, rgba(245,158,11,0.35));'></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    _, c_casino, _ = st.columns([2, 3, 2])
    with c_casino:
        with st.container(border=True):
            with st.popover("🎰  Open Roulette Table", use_container_width=True):
                render_roulette()

    # ── Bottom bar: Start New Game + API status ───────────────────────────────
    st.markdown('<div class="ctrl-bottom-border" style="margin-top:0.9rem;"></div>',
                unsafe_allow_html=True)

    c_start, _, c_api = st.columns([2, 3, 2])

    with c_start:
        if st.button("🎲  Start New Game", type="primary", use_container_width=True):
            with st.spinner("Fetching fresh S&P 500 data …"):
                try:
                    db.reset_all(INITIAL_CAPITAL)
                    fresh_df = md.fetch_sp500_prices_csv(CSV_PATH)
                    md.fetch_and_store_sp500(db)
                    st.cache_data.clear()
                except Exception as exc:
                    st.error(f"Failed to fetch market data: {exc}")
                    st.stop()

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
                "pending_day_data": None,
                "auto_deal":        True,
            })
            st.rerun()

    with c_api:
        if API_KEY:
            st.success("Claude AI · connected", icon="🤖")
        else:
            st.warning("Rule-based mode", icon="⚙️")

    st.markdown('<div class="ctrl-bottom-border"></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ── Main tabs
    # ─────────────────────────────────────────────────────────────────────────
    tab_summary, tab_portfolio, tab_market = st.tabs([
        "🗓️  Daily Summary",
        "📊  Portfolio",
        "🌐  Market Data",
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
        render_market_data(db)


if __name__ == "__main__":
    main()

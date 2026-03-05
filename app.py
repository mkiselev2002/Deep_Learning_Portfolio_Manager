"""
app.py  –  AI Paper Trading Simulator
══════════════════════════════════════
Single-page Streamlit app that runs a daily AI-driven investment loop over a
user-selected 1-week period from the synthetic ETF dataset.

Run:
    streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents import DataAgent, StrategyAgent, RiskAgent, ExecutionAgent
from config import (
    API_KEY,
    CSV_PATH,
    ETF_TICKERS,
    INITIAL_CAPITAL,
    LOOKBACK_DAYS,
    MAX_POSITION_PCT,
    MAX_TRADES_PER_DAY,
    SIMULATION_DAYS,
)
from database import PortfolioDatabase
import market_data as md

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Paper Trading Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom styling ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
    .stMetric label {font-size: 0.78rem; color: #888;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading ETF price data …")
def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def get_sim_dates(prices_df: pd.DataFrame, start_date, n: int = SIMULATION_DAYS) -> list:
    """Return up to *n* actual trading days from the CSV starting on or after start_date."""
    ts = pd.Timestamp(start_date)
    return prices_df.index[prices_df.index >= ts][:n].tolist()


def ensure_data_exists():
    """Run generate_data.py if the CSV is missing."""
    if not Path(CSV_PATH).exists():
        st.info("Generating synthetic ETF price data (runs once) …")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "generate_data.py")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            st.error(f"Data generation failed:\n{result.stderr}")
            st.stop()
        st.cache_data.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    prices_df:       pd.DataFrame,
    sim_dates:       list,
    strategy:        str,
    initial_capital: float,
    db:              PortfolioDatabase,
    progress_bar,
    status_box,
) -> dict[str, Any]:
    """
    Runs the full 5-day simulation.

    Returns a results dict consumed by the display layer.
    """
    db.reset(initial_capital)

    data_agent      = DataAgent()
    strategy_agent  = StrategyAgent(strategy)
    risk_agent      = RiskAgent()
    execution_agent = ExecutionAgent(db)

    results: dict[str, Any] = {"days": []}

    for i, sim_date in enumerate(sim_dates):
        day_num  = i + 1
        date_str = sim_date.strftime("%Y-%m-%d")

        status_box.markdown(
            f"**Day {day_num}/{len(sim_dates)}  –  {date_str}**  \n"
            f"Running agents …"
        )

        prices    = prices_df.loc[sim_date].to_dict()
        portfolio = db.get_portfolio_state(prices)

        # ── Step 1: Data Agent — compute technical indicators ────────────
        analysis = data_agent.analyze(prices_df, sim_date)

        # ── Step 2: Strategy Agent — propose trades ──────────────────────
        proposed = strategy_agent.propose_trades(analysis, portfolio, date_str, day_num)

        # ── Step 3: Risk Agent — validate & trim ─────────────────────────
        approved = risk_agent.validate(proposed, portfolio, prices)

        # ── Step 4: Execution Agent — commit to DB ───────────────────────
        exec_report = execution_agent.execute(approved, prices, date_str)

        results["days"].append({
            "date":            date_str,
            "day_num":         day_num,
            "portfolio":       exec_report["portfolio"],
            "analysis":        analysis,
            "proposed_trades": proposed,
            "approved_trades": exec_report["executed_trades"],
            "violations":      risk_agent.violations.copy(),
            "llm_reasoning":   strategy_agent.reasoning,
        })

        progress_bar.progress((i + 1) / len(sim_dates))

    status_box.empty()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plotly charts
# ═══════════════════════════════════════════════════════════════════════════════

_DARK = "plotly_dark"

def chart_portfolio_value(history: pd.DataFrame, initial_capital: float) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history["date"], y=history["total_equity"],
        name="Total Equity",
        mode="lines+markers",
        line=dict(color="#7c3aed", width=3),
        marker=dict(size=9),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
    ))
    fig.add_hline(
        y=initial_capital,
        line_dash="dot", line_color="gray",
        annotation_text=f"Initial  ${initial_capital:,.0f}",
        annotation_position="bottom right",
        annotation_font_color="gray",
    )
    fig.update_layout(
        title="Portfolio Value Over the Simulation Week",
        xaxis_title="Date",
        yaxis_title="Value (USD)",
        template=_DARK,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
        margin=dict(t=50, b=30),
    )
    return fig


def chart_allocation(portfolio: dict, total_value: float) -> go.Figure:
    labels = ["Cash"] + list(portfolio["positions"].keys())
    values = [portfolio["cash"]] + [p["value"] for p in portfolio["positions"].values()]
    colors = ["#06b6d4"] + px.colors.qualitative.Vivid

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.42,
        marker_colors=colors[:len(labels)],
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig.update_layout(
        title="Final Allocation",
        template=_DARK,
        height=350,
        showlegend=False,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def chart_signals(analysis: dict[str, dict], strategy: str) -> go.Figure:
    tickers  = list(analysis.keys())
    col      = "momentum_20d" if strategy == "Momentum" else "zscore"
    label    = "20-Day Momentum (%)" if strategy == "Momentum" else "Mean-Rev Z-Score"
    values   = [analysis[t][col] for t in tickers]
    colors   = ["#10b981" if v > 0 else "#ef553b" for v in values]

    fig = go.Figure(go.Bar(
        x=tickers, y=values,
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
    ))
    ref_line = 1.0 if strategy == "Mean Reversion" else 0.0
    fig.add_hline(y=ref_line, line_dash="dot", line_color="gray")
    if strategy == "Mean Reversion":
        fig.add_hline(y=-ref_line, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"{label} — Day Signals",
        yaxis_title=label,
        template=_DARK,
        height=280,
        margin=dict(t=50, b=30),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════════════

def render_sp500_tab(db: PortfolioDatabase):
    """Renders the S&P 500 Market Data tab."""
    st.subheader("S&P 500 Market Data")
    st.caption(
        "Fetches all 503 S&P 500 constituents with latest OHLCV prices via yfinance. "
        "Data is cached in the local database."
    )

    fetch_dates = db.get_sp500_fetch_dates()
    col_btn, col_date = st.columns([1, 3])

    with col_btn:
        fetch_btn = st.button("Fetch / Refresh Data", type="primary", use_container_width=True)

    if fetch_btn:
        with st.spinner("Downloading S&P 500 prices from Yahoo Finance (may take ~30 s) …"):
            try:
                md.fetch_and_store_sp500(db)
                st.success("Data refreshed successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to fetch data: {exc}")
                return

    # Re-read dates after potential fetch
    fetch_dates = db.get_sp500_fetch_dates()

    if not fetch_dates:
        st.info("No data yet — press **Fetch / Refresh Data** to download today's S&P 500 prices.")
        return

    with col_date:
        selected_date = st.selectbox(
            "Snapshot date",
            options=fetch_dates,
            index=0,
            label_visibility="collapsed",
        )

    df = db.get_sp500_stocks(selected_date)
    if df.empty:
        st.warning("No data found for the selected date.")
        return

    # ── Computed columns ──────────────────────────────────────────────────────
    df["change_pct"] = ((df["close"] - df["open"]) / df["open"] * 100).round(2)

    # ── Summary metrics ───────────────────────────────────────────────────────
    total = len(df)
    gainers = int((df["change_pct"] > 0).sum())
    losers  = int((df["change_pct"] < 0).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Stocks",  str(total))
    m2.metric("Gainers",       str(gainers),  delta=f"+{gainers}")
    m3.metric("Losers",        str(losers),   delta=f"-{losers}", delta_color="inverse")
    m4.metric("Snapshot Date", selected_date)

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("Search by symbol or company name", placeholder="e.g. AAPL or Apple")
    with fc2:
        sectors = ["All Sectors"] + sorted(df["sector"].dropna().unique().tolist())
        sector_filter = st.selectbox("Sector", options=sectors)
    with fc3:
        direction = st.selectbox("Direction", ["All", "Gainers only", "Losers only"])

    filtered = df.copy()
    if search:
        q = search.upper()
        filtered = filtered[
            filtered["symbol"].str.upper().str.contains(q, na=False)
            | filtered["name"].str.upper().str.contains(q, na=False)
        ]
    if sector_filter != "All Sectors":
        filtered = filtered[filtered["sector"] == sector_filter]
    if direction == "Gainers only":
        filtered = filtered[filtered["change_pct"] > 0]
    elif direction == "Losers only":
        filtered = filtered[filtered["change_pct"] < 0]

    st.caption(f"Showing {len(filtered)} of {total} stocks")

    # ── Display table ─────────────────────────────────────────────────────────
    display = filtered[["symbol", "name", "sector", "open", "high", "low", "close", "change_pct", "volume"]].copy()
    display.columns = ["Symbol", "Company", "Sector", "Open", "High", "Low", "Close", "Change %", "Volume"]

    def _style_change(val):
        if pd.isna(val):
            return ""
        return "color:#10b981; font-weight:bold" if val > 0 else ("color:#ef553b; font-weight:bold" if val < 0 else "")

    styled = (
        display.style
        .map(_style_change, subset=["Change %"])
        .format({
            "Open":     lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "High":     lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "Low":      lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "Close":    lambda v: f"${v:.2f}" if pd.notna(v) else "—",
            "Change %": lambda v: f"{v:+.2f}%" if pd.notna(v) else "—",
            "Volume":   lambda v: f"{int(v):,}" if pd.notna(v) else "—",
        })
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)


def main():
    ensure_data_exists()

    prices_df = load_price_data()
    db        = PortfolioDatabase()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("AI Paper Trader")
        st.caption("Configure your simulation below.")

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10_000,
            max_value=10_000_000,
            value=int(INITIAL_CAPITAL),
            step=10_000,
        )

        strategy = st.selectbox(
            "Strategy",
            ["Momentum", "Mean Reversion"],
            help=(
                "**Momentum** – buy recent winners, sell laggards.\n\n"
                "**Mean Reversion** – buy oversold ETFs (low z-score), "
                "sell overbought ones."
            ),
        )

        st.divider()

        min_date     = prices_df.index[LOOKBACK_DAYS].date()
        max_date     = prices_df.index[-SIMULATION_DAYS - 1].date()
        default_date = prices_df.index[-SIMULATION_DAYS - 5].date()

        sim_start = st.date_input(
            "Simulation Week Start",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
            help="Choose any weekday.  The simulator will find the next 5 trading days.",
        )
        sim_dates = get_sim_dates(prices_df, sim_start)

        if len(sim_dates) < SIMULATION_DAYS:
            st.error(f"Need {SIMULATION_DAYS} trading days after {sim_start}. Choose an earlier date.")
            sim_dates = []
        else:
            st.info(
                f"{sim_dates[0].strftime('%b %d')} → {sim_dates[-1].strftime('%b %d, %Y')}"
            )

        st.divider()
        st.caption("**Risk Rules (enforced)**")
        st.markdown(
            f"- Max position weight: **{MAX_POSITION_PCT*100:.0f}%**\n"
            f"- Max trades per day: **{MAX_TRADES_PER_DAY}**"
        )
        st.divider()

        if API_KEY:
            st.success("Claude API: connected", icon="🤖")
        else:
            st.warning("No API key found – rule-based fallback mode", icon="⚙️")

        run_btn   = st.button("Run Simulation", type="primary", use_container_width=True)
        reset_btn = st.button("Reset",                           use_container_width=True)

    # ── Page header ──────────────────────────────────────────────────────────
    st.title("AI Paper Trading Simulator")
    st.markdown(
        "Three AI agents collaborate every trading day: **Market Analysis** → "
        "**Trade Proposal** (Claude LLM) → **Risk Validation**."
    )

    tab_sim, tab_market = st.tabs(["Paper Trading Simulation", "S&P 500 Market Data"])

    with tab_market:
        render_sp500_tab(db)

    with tab_sim:
        # ── Reset action ──────────────────────────────────────────────────────
        if reset_btn:
            db.reset(initial_capital)
            st.session_state.pop("sim_results", None)
            st.session_state.pop("sim_capital", None)
            st.rerun()

        # ── Run simulation ────────────────────────────────────────────────────
        if run_btn and sim_dates:
            st.markdown("---")
            progress_bar = st.progress(0, "Initialising …")
            status_box   = st.empty()

            try:
                results = run_simulation(
                    prices_df, sim_dates, strategy,
                    initial_capital, db, progress_bar, status_box,
                )
            except Exception as exc:
                progress_bar.empty()
                status_box.error(f"Simulation error: {exc}")
                st.stop()

            progress_bar.empty()
            st.session_state["sim_results"]  = results
            st.session_state["sim_capital"]  = initial_capital
            st.session_state["sim_strategy"] = strategy
            st.session_state["sim_dates"]    = [d.strftime("%Y-%m-%d") for d in sim_dates]
            st.rerun()

        # ── Results display ───────────────────────────────────────────────────
        history_df = db.get_portfolio_history()

        if history_df.empty:
            st.info("Configure the parameters in the sidebar and press **Run Simulation**.")

            st.subheader("ETF Universe — Latest Prices")
            latest = prices_df[ETF_TICKERS].tail(10).copy()
            st.dataframe(
                latest.style.format("${:.2f}"),
                use_container_width=True,
                height=320,
            )

        else:
            # ── KPI row ───────────────────────────────────────────────────────
            init_cap     = st.session_state.get("sim_capital", initial_capital)
            strategy_lbl = st.session_state.get("sim_strategy", strategy)

            final_val  = history_df["total_equity"].iloc[-1]
            pnl        = final_val - init_cap
            pnl_pct    = pnl / init_cap * 100.0
            max_val    = history_df["total_equity"].max()
            drawdown   = (max_val - history_df["total_equity"].min()) / max_val * 100.0
            num_trades = len(db.get_trades())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Value",     f"${final_val:,.2f}",  delta=f"${pnl:+,.2f}")
            c2.metric("Total Return",    f"{pnl_pct:+.2f}%")
            c3.metric("Peak Value",      f"${max_val:,.2f}")
            c4.metric("Max Drawdown",    f"{drawdown:.2f}%")
            c5.metric("Trades Executed", str(num_trades))

            st.divider()

            # ── Main charts ───────────────────────────────────────────────────
            col_main, col_side = st.columns([2, 1])

            with col_main:
                fig_val = chart_portfolio_value(history_df, init_cap)
                st.plotly_chart(fig_val, use_container_width=True)

            with col_side:
                results       = st.session_state.get("sim_results")
                sim_date_strs = st.session_state.get("sim_dates", [])

                if results and sim_date_strs:
                    last_date    = pd.Timestamp(sim_date_strs[-1])
                    final_prices = prices_df.loc[last_date].to_dict()
                    final_port   = db.get_portfolio_state(final_prices)
                    fig_alloc    = chart_allocation(final_port, final_val)
                    st.plotly_chart(fig_alloc, use_container_width=True)

            st.divider()

            # ── Trade history ─────────────────────────────────────────────────
            st.subheader("Trade History")
            trades_df = db.get_trades()

            if trades_df.empty:
                st.info("No trades were executed during this simulation.")
            else:
                def _style_action(val):
                    return "color:#10b981; font-weight:bold" if val == "BUY" else "color:#ef553b; font-weight:bold"

                display = trades_df[["date", "symbol", "action", "shares", "price", "reason"]].copy()
                display["price"] = display["price"].map("${:.2f}".format)
                st.dataframe(
                    display.style.map(_style_action, subset=["action"]),
                    use_container_width=True,
                    hide_index=True,
                )

            st.divider()

            # ── Day-by-day expanders ──────────────────────────────────────────
            if results:
                st.subheader("Daily Agent Activity")

                for day in results["days"]:
                    badge = f"✅  {len(day['approved_trades'])} trade(s)" if day["approved_trades"] else "— no trades"
                    with st.expander(f"Day {day['day_num']}  •  {day['date']}  •  {badge}"):
                        left, right = st.columns([3, 2])

                        with left:
                            st.markdown("**Market Analysis**")
                            rows = []
                            for t, m in day["analysis"].items():
                                rows.append({
                                    "ETF":       t,
                                    "Price":     f"${m['price']:.2f}",
                                    "Mom 20d":   f"{m['momentum_20d']:+.2f}%",
                                    "Mom 5d":    f"{m['momentum_5d']:+.2f}%",
                                    "Z-Score":   f"{m['zscore']:+.3f}",
                                    "RSI":       f"{m['rsi']:.1f}",
                                    "vs SMA20":  "▲" if m["above_sma20"] else "▼",
                                })
                            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                            st.plotly_chart(
                                chart_signals(day["analysis"], strategy_lbl),
                                use_container_width=True,
                            )

                        with right:
                            st.markdown("**Trade Proposals & Decisions**")
                            approved_keys = {
                                (t["ticker"], t["action"]) for t in day["approved_trades"]
                            }
                            for prop in day["proposed_trades"]:
                                key     = (prop["ticker"], prop["action"])
                                icon    = "🟢" if prop["action"] == "BUY" else "🔴"
                                verdict = "✅ approved" if key in approved_keys else "❌ rejected"
                                st.markdown(
                                    f"{icon} **{prop['action']} {prop['ticker']}** &nbsp; {verdict}"
                                )
                                st.caption(prop.get("reasoning", ""))

                            if not day["proposed_trades"]:
                                st.info("No trades proposed today.")

                            if day["violations"]:
                                st.markdown("**Risk Agent messages**")
                                for v in day["violations"]:
                                    st.warning(v, icon="⚠️")

                            st.markdown("**End-of-Day Portfolio**")
                            snap = day["portfolio"]
                            port_rows = [{"Item": "Cash", "Value": f"${snap['cash']:,.2f}", "Weight": "—"}]
                            for t, pos in snap["positions"].items():
                                w = pos["value"] / snap["total_value"] * 100
                                port_rows.append({
                                    "Item":   t,
                                    "Value":  f"${pos['value']:,.2f}",
                                    "Weight": f"{w:.1f}%",
                                })
                            port_rows.append({
                                "Item":   "TOTAL",
                                "Value":  f"${snap['total_value']:,.2f}",
                                "Weight": "100%",
                            })
                            st.dataframe(pd.DataFrame(port_rows), hide_index=True, use_container_width=True)

                            if day.get("llm_reasoning"):
                                with st.popover("View raw LLM output"):
                                    st.code(day["llm_reasoning"], language="json")

            st.divider()

            # ── Final positions ───────────────────────────────────────────────
            st.subheader("Final Positions")
            sim_date_strs2 = st.session_state.get("sim_dates", [])
            if sim_date_strs2:
                last_date2    = pd.Timestamp(sim_date_strs2[-1])
                final_prices2 = prices_df.loc[last_date2].to_dict()
                final_port2   = db.get_portfolio_state(final_prices2)

                if final_port2["positions"]:
                    rows = []
                    for t, p in final_port2["positions"].items():
                        rows.append({
                            "ETF":           t,
                            "Shares":        p["shares"],
                            "Avg Cost":      f"${p['average_cost']:.2f}",
                            "Current Price": f"${p['current_price']:.2f}",
                            "Market Value":  f"${p['value']:,.2f}",
                            "Weight":        f"{p['value']/final_port2['total_value']*100:.1f}%",
                            "P&L":           f"{p['pnl_pct']:+.2f}%",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.info(f"No open positions.  Cash: ${final_port2['cash']:,.2f}")

            st.caption(
                "Simulation uses synthetic ETF data (correlated GBM model).  "
                "AI proposals are generated by Claude (Anthropic).  Not financial advice."
            )


if __name__ == "__main__":
    main()

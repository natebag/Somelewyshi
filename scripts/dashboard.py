"""Streamlit dashboard for Miro Fish — live monitoring & trade history.

Run with:  streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

from db.engine import get_engine, get_session
from db.models import PipelineRun, PipelineStatus, Position, PositionStatus, Trade, TradeStatus
from sqlmodel import select, func
from trading.positions import PositionTracker

# ── Page Config ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Miro Fish",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ───────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e0e0e0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .profit { color: #00d26a !important; }
    .loss { color: #ff4757 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────

@st.cache_resource
def get_db():
    engine = get_engine()
    return engine


def load_data():
    engine = get_db()
    session = get_session(engine)

    trades = session.exec(
        select(Trade).order_by(Trade.created_at.desc()).limit(100)
    ).all()

    runs = session.exec(
        select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(50)
    ).all()

    tracker = PositionTracker(session)
    portfolio = tracker.get_portfolio_summary()
    open_positions = tracker.get_open_positions()
    all_positions = tracker.get_all_positions()

    return trades, runs, portfolio, open_positions, all_positions, session


# ── Header ───────────────────────────────────────────────────────

st.markdown("# 🐟 Miro Fish Dashboard")
st.markdown("*Predict · Trade · Broadcast*")
st.divider()

trades, runs, portfolio, open_positions, all_positions, session = load_data()

# ── Top Metrics Row ──────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    pnl_class = "profit" if portfolio.total_realized_pnl >= 0 else "loss"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {pnl_class}">${portfolio.total_realized_pnl:+.2f}</div>
        <div class="metric-label">Realized P&L</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    upnl_class = "profit" if portfolio.total_unrealized_pnl >= 0 else "loss"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {upnl_class}">${portfolio.total_unrealized_pnl:+.2f}</div>
        <div class="metric-label">Unrealized P&L</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{portfolio.win_rate:.0%}</div>
        <div class="metric-label">Win Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(trades)}</div>
        <div class="metric-label">Total Trades</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{portfolio.open_positions}</div>
        <div class="metric-label">Open Positions</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ── Charts Row ───────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Trade EV Distribution")
    if trades:
        evs = [t.expected_value for t in trades if t.expected_value]
        if evs:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=evs,
                nbinsx=20,
                marker_color="#6c5ce7",
                opacity=0.8,
            ))
            fig.update_layout(
                xaxis_title="EV per Dollar",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade EV data yet.")
    else:
        st.info("No trades yet. Run the bot to see data here.")

with chart_col2:
    st.subheader("Cumulative P&L")
    if trades:
        # Build cumulative P&L from trades (simplified — uses EV as proxy)
        sorted_trades = sorted(trades, key=lambda t: t.created_at)
        cumulative = []
        running = 0
        times = []
        for t in sorted_trades:
            if t.expected_value and t.size_usd:
                running += t.expected_value * t.size_usd
                cumulative.append(running)
                times.append(t.created_at)

        if cumulative:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times,
                y=cumulative,
                mode="lines+markers",
                line=dict(color="#00d26a" if cumulative[-1] >= 0 else "#ff4757", width=2),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor="rgba(0,210,106,0.1)" if cumulative[-1] >= 0 else "rgba(255,71,87,0.1)",
            ))
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Expected P&L ($)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No P&L data yet.")
    else:
        st.info("No trades yet.")

# ── Open Positions ───────────────────────────────────────────────

st.subheader("Open Positions")
if open_positions:
    pos_data = []
    for p in open_positions:
        pnl_str = f"${p.unrealized_pnl:+.2f}"
        pos_data.append({
            "Market": p.market_id[:40] if p.market_id else "N/A",
            "Side": p.side,
            "Entry": f"{p.entry_price:.0%}",
            "Current": f"{p.current_price:.0%}",
            "Size": f"${p.size_usd:.2f}",
            "P&L": pnl_str,
            "Log Return": f"{p.log_return:+.4f}",
        })
    st.dataframe(pos_data, use_container_width=True, hide_index=True)
else:
    st.info("No open positions.")

# ── Recent Trades ────────────────────────────────────────────────

st.subheader("Recent Trades")
if trades:
    trade_data = []
    for t in trades[:20]:
        trade_data.append({
            "Time": t.created_at.strftime("%Y-%m-%d %H:%M"),
            "Side": t.side,
            "Price": f"{t.market_price_at_entry:.0%}",
            "EV": f"{t.expected_value:.1%}" if t.expected_value else "—",
            "Kelly": f"{t.kelly_fraction:.1%}" if t.kelly_fraction else "—",
            "Size": f"${t.size_usd:.2f}",
            "Status": t.status.value,
        })
    st.dataframe(trade_data, use_container_width=True, hide_index=True)
else:
    st.info("No trades yet. Run `mirofish run` or `mirofish daemon` to start.")

# ── Recent Pipeline Runs ────────────────────────────────────────

st.subheader("Recent Pipeline Runs")
if runs:
    run_data = []
    for r in runs[:10]:
        run_data.append({
            "ID": r.id[:8],
            "Market": r.market_query[:50],
            "Status": r.status.value,
            "Started": r.started_at.strftime("%Y-%m-%d %H:%M:%S"),
        })
    st.dataframe(run_data, use_container_width=True, hide_index=True)
else:
    st.info("No pipeline runs yet.")

# ── Sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### System Info")
    st.markdown(f"**Log Return:** `{portfolio.total_log_return:+.4f}`")
    st.markdown(f"**Total Invested:** `${portfolio.total_invested:.2f}`")
    st.markdown(f"**Closed Positions:** `{portfolio.closed_positions}`")

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

session.close()

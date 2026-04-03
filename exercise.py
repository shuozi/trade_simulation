"""
exercise.py — Manual Trading Journal & Strategy Exercise Dashboard
==================================================================
A Dash UI to practice strategies manually.

Features:
  • Trade Entry  — log any buy/sell with date, price, quantity
  • Portfolio    — live P&L for open positions (fetches current prices)
  • Signals      — strategy signals for each holding from strategy.py
  • Performance  — equity curve, Sharpe, win rate, max drawdown, avg hold
  • Trade Log    — full history with realized P&L per closed trade
  • Analysis     — per-asset breakdown + best/worst trades

Data is saved to trades.json (persistent between sessions).

Usage:
  python exercise.py              # start on http://localhost:8051
  python exercise.py --port 8888  # custom port
  python exercise.py --reset      # clear all trades and start fresh
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

TRADES_FILE = "trades.json"


def load_trades() -> list[dict]:
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_trades(trades: list[dict]):
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def next_id(trades: list[dict]) -> int:
    return max((t.get("id", 0) for t in trades), default=0) + 1


# ─────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────

_price_cache: dict[str, float] = {}
_history_cache: dict[str, pd.DataFrame] = {}


def get_current_price(symbol: str) -> Optional[float]:
    if symbol in _price_cache:
        return _price_cache[symbol]
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        hist = tk.history(period="2d")
        if hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])
        _price_cache[symbol] = price
        return price
    except Exception:
        return None


def get_history(symbol: str, days: int = 365) -> pd.DataFrame:
    if symbol in _history_cache:
        return _history_cache[symbol]
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        _history_cache[symbol] = df
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# STRATEGY SIGNALS (inline, no import needed)
# ─────────────────────────────────────────────

def _sma(c, w): return c.rolling(w).mean()
def _ema(c, sp): return c.ewm(span=sp, adjust=False).mean()
def _rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def run_signals(df: pd.DataFrame) -> dict:
    """Return dict of signal_name → (vote, label)."""
    if df.empty or len(df) < 22:
        return {}
    c   = df["close"].astype(float)
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)
    out = {}

    # Trend
    if len(c) >= 50:
        s50  = _sma(c, 50).iloc[-1]
        s50p = _sma(c, 50).iloc[-2]
        suf  = f"SMA50={s50:.2f}"
        if c.iloc[-1] > s50 and c.iloc[-2] <= s50p:
            out["Trend"] = (1, "Price crossed above SMA50 ↑")
        elif c.iloc[-1] > s50:
            out["Trend"] = (1, f"Price > {suf}")
        elif c.iloc[-1] < s50:
            out["Trend"] = (-1, f"Price < {suf}")
        else:
            out["Trend"] = (0, "At SMA50")

    # Momentum 12-1
    n = min(252, len(c) - 1)
    skip = min(21, len(c) - 2)
    if len(c) > skip + 2:
        mom = (c.iloc[-(skip+1)] / c.iloc[-n]) - 1
        if mom > 0.10:   out["Momentum"] = (1,  f"12-1mo ret: +{mom*100:.1f}%")
        elif mom < -0.10: out["Momentum"] = (-1, f"12-1mo ret: {mom*100:.1f}%")
        else:             out["Momentum"] = (0,  f"12-1mo ret: {mom*100:.1f}%")

    # RSI
    rsi = float(_rsi(c).iloc[-1])
    if rsi < 30:   out["RSI"] = (1,  f"RSI={rsi:.1f} oversold")
    elif rsi > 70: out["RSI"] = (-1, f"RSI={rsi:.1f} overbought")
    elif rsi < 45: out["RSI"] = (1,  f"RSI={rsi:.1f} bullish zone")
    elif rsi > 55: out["RSI"] = (-1, f"RSI={rsi:.1f} bearish zone")
    else:          out["RSI"] = (0,  f"RSI={rsi:.1f} neutral")

    # MACD
    if len(c) >= 35:
        ml = _ema(c, 12) - _ema(c, 26)
        ms = ml.ewm(span=9, adjust=False).mean()
        if ml.iloc[-2] <= ms.iloc[-2] and ml.iloc[-1] > ms.iloc[-1]:
            out["MACD"] = (1, "Bullish crossover ↑")
        elif ml.iloc[-2] >= ms.iloc[-2] and ml.iloc[-1] < ms.iloc[-1]:
            out["MACD"] = (-1, "Bearish crossover ↓")
        elif ml.iloc[-1] > ms.iloc[-1]:
            out["MACD"] = (1, "MACD above signal")
        else:
            out["MACD"] = (-1, "MACD below signal")

    # Bollinger
    mid = c.rolling(20).mean()
    sd  = c.rolling(20).std()
    up, lo = mid + 2*sd, mid - 2*sd
    p = c.iloc[-1]
    if p < lo.iloc[-1]:   out["Bollinger"] = (1,  "Below lower band — oversold")
    elif p > up.iloc[-1]: out["Bollinger"] = (-1, "Above upper band — overbought")
    elif (p - lo.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1] + 1e-10) < 0.35:
        out["Bollinger"] = (1,  "Near lower band")
    elif (p - lo.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1] + 1e-10) > 0.65:
        out["Bollinger"] = (-1, "Near upper band")
    else:
        out["Bollinger"] = (0,  "Mid band — neutral")

    # Volume surge
    if not vol.empty and len(vol) >= 21:
        vz = (vol.iloc[-1] - vol.rolling(20).mean().iloc[-1]) / (vol.rolling(20).std().iloc[-1] + 1e-10)
        ret = float(c.pct_change().iloc[-1])
        if vz > 1.5 and ret > 0:   out["Volume"] = (1,  f"Volume surge {vz:.1f}σ ↑")
        elif vz > 1.5 and ret < 0: out["Volume"] = (-1, f"Volume surge {vz:.1f}σ ↓")
        else:                       out["Volume"] = (0,  f"Volume normal {vz:.1f}σ")

    # Short-term reversal
    r5 = float((c.iloc[-1] / c.iloc[min(-5, -(len(c)-1))]) - 1)
    rsi7 = float(_rsi(c, 7).iloc[-1]) if len(c) >= 9 else 50
    if r5 < -0.05 and rsi7 < 35:   out["Reversal"] = (1,  f"5d {r5*100:.1f}% + RSI={rsi7:.0f} bounce")
    elif r5 > 0.05 and rsi7 > 65:  out["Reversal"] = (-1, f"5d {r5*100:.1f}% + RSI={rsi7:.0f} fade")
    else:                           out["Reversal"] = (0,  f"5d ret={r5*100:.1f}%")

    return out


def signal_summary(sigs: dict) -> tuple[str, str]:
    """Derive overall signal label and color from individual votes."""
    if not sigs:
        return "NO DATA", "#555"
    votes = [v for v, _ in sigs.values()]
    score = sum(votes)
    if score >= 3:    return "STRONG BUY",  "#27ae60"
    if score == 2:    return "BUY",          "#2ecc71"
    if score == 1:    return "WEAK BUY",     "#a8e6cf"
    if score == -1:   return "WEAK SELL",    "#f39c12"
    if score == -2:   return "SELL",         "#e67e22"
    if score <= -3:   return "STRONG SELL",  "#e74c3c"
    return "NEUTRAL", "#95a5a6"


# ─────────────────────────────────────────────
# PORTFOLIO CALCULATION
# ─────────────────────────────────────────────

def compute_portfolio(trades: list[dict]) -> dict:
    """
    From raw trade list, compute:
      - open_positions: {symbol: {qty, avg_cost, current_price, unrealized_pnl, ...}}
      - closed_trades:  list of matched buy→sell with realized P&L
      - equity_curve:   daily portfolio value timeseries
      - summary metrics
    """
    if not trades:
        return {"open": {}, "closed": [], "equity": pd.Series(dtype=float),
                "total_invested": 0, "total_value": 0,
                "unrealized_pnl": 0, "realized_pnl": 0}

    df = pd.DataFrame(trades)
    df["date"]     = pd.to_datetime(df["date"])
    df["price"]    = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype(float)
    df["value"]    = df["price"] * df["quantity"]

    # Open positions (FIFO matching)
    open_pos = {}
    closed   = []

    for sym, grp in df.groupby("symbol"):
        buys  = []   # (date, price, qty) FIFO queue
        for _, row in grp.sort_values("date").iterrows():
            if row["side"] == "BUY":
                buys.append({"date": row["date"], "price": row["price"],
                              "qty": row["quantity"]})
            else:  # SELL — match against oldest buys
                sell_qty   = row["quantity"]
                sell_price = row["price"]
                while sell_qty > 0 and buys:
                    lot = buys[0]
                    matched = min(lot["qty"], sell_qty)
                    pnl = (sell_price - lot["price"]) * matched
                    closed.append({
                        "symbol":       sym,
                        "buy_date":     lot["date"].strftime("%Y-%m-%d"),
                        "sell_date":    row["date"].strftime("%Y-%m-%d"),
                        "qty":          matched,
                        "buy_price":    lot["price"],
                        "sell_price":   sell_price,
                        "realized_pnl": round(pnl, 2),
                        "return_pct":   round((sell_price / lot["price"] - 1) * 100, 2),
                        "hold_days":    (row["date"] - lot["date"]).days,
                    })
                    lot["qty"] -= matched
                    sell_qty   -= matched
                    if lot["qty"] <= 0:
                        buys.pop(0)

        # Remaining buys = open position
        if buys:
            total_qty  = sum(b["qty"] for b in buys)
            avg_cost   = sum(b["price"] * b["qty"] for b in buys) / total_qty
            open_pos[sym] = {
                "symbol":       sym,
                "quantity":     total_qty,
                "avg_cost":     round(avg_cost, 4),
                "open_date":    buys[0]["date"].strftime("%Y-%m-%d"),
                "invested":     round(avg_cost * total_qty, 2),
            }

    # Fetch current prices for open positions
    total_value      = 0.0
    total_invested   = 0.0
    unrealized_total = 0.0

    for sym, pos in open_pos.items():
        cp = get_current_price(sym)
        pos["current_price"]   = round(cp, 4) if cp else pos["avg_cost"]
        pos["current_value"]   = round(pos["current_price"] * pos["quantity"], 2)
        pos["unrealized_pnl"]  = round((pos["current_price"] - pos["avg_cost"]) * pos["quantity"], 2)
        pos["return_pct"]      = round((pos["current_price"] / pos["avg_cost"] - 1) * 100, 2)
        pos["hold_days"]       = (datetime.today() - pd.to_datetime(pos["open_date"])).days
        total_value      += pos["current_value"]
        total_invested   += pos["invested"]
        unrealized_total += pos["unrealized_pnl"]

    realized_total = sum(c["realized_pnl"] for c in closed)

    return {
        "open":            open_pos,
        "closed":          closed,
        "total_invested":  round(total_invested, 2),
        "total_value":     round(total_value, 2),
        "unrealized_pnl":  round(unrealized_total, 2),
        "realized_pnl":    round(realized_total, 2),
        "total_pnl":       round(unrealized_total + realized_total, 2),
    }


def compute_performance(pf: dict) -> dict:
    """Compute Sharpe, win rate, best/worst trade from portfolio data."""
    closed = pf.get("closed", [])
    if not closed:
        return {"sharpe": None, "win_rate": None, "avg_return": None,
                "best_trade": None, "worst_trade": None, "avg_hold_days": None}
    rets = [c["return_pct"] for c in closed]
    wins = sum(1 for r in rets if r > 0)
    r_arr = np.array(rets)
    sharpe = float(r_arr.mean() / (r_arr.std() + 1e-10)) if len(r_arr) > 1 else 0
    best  = max(closed, key=lambda c: c["return_pct"])
    worst = min(closed, key=lambda c: c["return_pct"])
    return {
        "sharpe":       round(sharpe, 3),
        "win_rate":     round(wins / len(closed) * 100, 1),
        "avg_return":   round(float(r_arr.mean()), 2),
        "total_trades": len(closed),
        "best_trade":   best,
        "worst_trade":  worst,
        "avg_hold_days":round(np.mean([c["hold_days"] for c in closed]), 1),
    }


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────

DARK = "#12121f"
CARD = "#1e1e2e"
BORDER = "#333"

def _fig_base(title, height=350):
    return dict(
        title=title, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font_color="white", margin=dict(t=45, b=30, l=50, r=20),
        height=height,
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER),
    )


def build_pnl_chart(pf: dict) -> go.Figure:
    """Bar chart: unrealized P&L per open position."""
    pos = list(pf["open"].values())
    if not pos:
        return go.Figure().update_layout(**_fig_base("Unrealized P&L per Position"))
    syms  = [p["symbol"]       for p in pos]
    pnls  = [p["unrealized_pnl"] for p in pos]
    rets  = [p["return_pct"]   for p in pos]
    colors = ["#27ae60" if p >= 0 else "#e74c3c" for p in pnls]
    fig = go.Figure([
        go.Bar(x=syms, y=pnls, marker_color=colors, name="P&L ($)",
               hovertemplate="<b>%{x}</b><br>P&L: $%{y:.2f}<br>Return: " +
               "<br>".join([f"{r:+.2f}%" for r in rets]) + "<extra></extra>"),
    ])
    for i, (s, r) in enumerate(zip(syms, rets)):
        fig.add_annotation(x=s, y=pnls[i], text=f"{r:+.1f}%",
                           showarrow=False, yshift=10 if pnls[i] >= 0 else -18,
                           font=dict(size=11, color="white"))
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    fig.update_layout(**_fig_base("Unrealized P&L per Position"))
    return fig


def build_allocation_pie(pf: dict) -> go.Figure:
    """Pie: portfolio allocation by current value."""
    pos = list(pf["open"].values())
    if not pos:
        return go.Figure().update_layout(**_fig_base("Portfolio Allocation"))
    vals  = [p["current_value"] for p in pos]
    syms  = [p["symbol"]        for p in pos]
    fig = go.Figure(go.Pie(
        labels=syms, values=vals, hole=0.4,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Set3[:len(syms)]),
        hovertemplate="<b>%{label}</b><br>$%{value:.2f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**_fig_base("Portfolio Allocation", height=300),
                      showlegend=False)
    return fig


def build_price_with_entry(symbol: str, entry_price: float, entry_date: str,
                            current_price: Optional[float]) -> go.Figure:
    """Price chart with entry line and current price."""
    df = get_history(symbol, days=365)
    if df.empty:
        return go.Figure().update_layout(**_fig_base(f"{symbol} — No Data"))

    c = df["close"].astype(float)
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    mid    = c.rolling(20).mean()
    std_   = c.rolling(20).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=c, mode="lines", name="Price",
                             line=dict(color="#4fc3f7", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=sma50,  mode="lines", name="SMA50",
                             line=dict(color="#f39c12", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=sma200, mode="lines", name="SMA200",
                             line=dict(color="#3498db", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=mid + 2*std_, mode="lines",
                             name="BB Upper", line=dict(color="rgba(180,130,255,0.4)", width=1),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=mid - 2*std_, mode="lines",
                             name="BB Lower", fill="tonexty",
                             fillcolor="rgba(180,130,255,0.05)",
                             line=dict(color="rgba(180,130,255,0.4)", width=1),
                             showlegend=False))

    # Entry line
    fig.add_hline(y=entry_price, line_dash="dash", line_color="#27ae60", line_width=2,
                  annotation_text=f"Entry ${entry_price:.2f}", annotation_position="left",
                  annotation_font_color="#27ae60")
    # Current price line
    if current_price:
        color = "#27ae60" if current_price >= entry_price else "#e74c3c"
        fig.add_hline(y=current_price, line_dash="dot", line_color=color, line_width=1.5,
                      annotation_text=f"Now ${current_price:.2f}",
                      annotation_position="right", annotation_font_color=color)
    # Entry date marker
    try:
        ed = pd.to_datetime(entry_date)
        ep = float(c.asof(ed)) if ed in c.index else entry_price
        fig.add_trace(go.Scatter(x=[ed], y=[entry_price], mode="markers",
                                 marker=dict(size=12, color="#27ae60", symbol="triangle-up"),
                                 name="Entry"))
    except Exception:
        pass

    fig.update_layout(**_fig_base(f"{symbol} — Price Chart", height=420),
                      legend=dict(bgcolor=CARD, font_size=11))
    return fig


def build_signal_bars(sigs: dict, symbol: str) -> go.Figure:
    """Horizontal bar chart of signal votes for one position."""
    if not sigs:
        return go.Figure().update_layout(**_fig_base(f"{symbol} — Signals"))
    names  = list(sigs.keys())
    votes  = [v for v, _ in sigs.values()]
    labels = [lbl for _, lbl in sigs.values()]
    colors = ["#27ae60" if v > 0 else ("#e74c3c" if v < 0 else "#555") for v in votes]
    fig = go.Figure(go.Bar(
        x=votes, y=names, orientation="h",
        marker_color=colors,
        text=labels, textposition="outside",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{y}: %{x:+d}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="white", line_width=0.5)
    fig.update_layout(**_fig_base(f"{symbol} — Strategy Signals", height=300),
                      xaxis=dict(tickvals=[-1, 0, 1], range=[-1.6, 2.4], gridcolor=BORDER),
                      margin=dict(t=40, b=20, l=90, r=160))
    return fig


def build_closed_pnl(closed: list[dict]) -> go.Figure:
    """Cumulative realized P&L + bar per trade."""
    if not closed:
        return go.Figure().update_layout(**_fig_base("Realized P&L — Trade History"))
    df = pd.DataFrame(closed).sort_values("sell_date")
    df["cumulative"] = df["realized_pnl"].cumsum()
    colors = ["#27ae60" if p >= 0 else "#e74c3c" for p in df["realized_pnl"]]
    labels = [f"{r['symbol']} {r['sell_date']}" for _, r in df.iterrows()]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=labels, y=df["cumulative"], mode="lines+markers",
                             name="Cumulative P&L", line=dict(color="#4fc3f7", width=2),
                             fill="tozeroy", fillcolor="rgba(79,195,247,0.1)"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=df["realized_pnl"], name="Trade P&L",
                         marker_color=colors), row=2, col=1)
    fig.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD, font_color="white",
                      height=420, margin=dict(t=40, b=60, l=50, r=20),
                      title="Realized P&L — Trade History",
                      legend=dict(bgcolor=CARD),
                      xaxis2=dict(gridcolor=BORDER, tickangle=-30, tickfont_size=9),
                      xaxis=dict(gridcolor=BORDER),
                      yaxis=dict(gridcolor=BORDER),
                      yaxis2=dict(gridcolor=BORDER))
    fig.add_hline(y=0, line_color="white", line_width=0.5, row=2, col=1)
    return fig


# ─────────────────────────────────────────────
# COMPONENT BUILDERS
# ─────────────────────────────────────────────

def card_metric(label, value, subtitle="", color="#fff"):
    return dbc.Card([dbc.CardBody([
        html.P(label, className="mb-0 text-muted", style={"fontSize": "0.8rem"}),
        html.H4(value, className="mb-0", style={"color": color, "fontWeight": "bold"}),
        html.P(subtitle, className="mb-0 text-muted", style={"fontSize": "0.75rem"}),
    ])], style={"background": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "8px", "textAlign": "center"})


def pnl_color(v: float) -> str:
    return "#27ae60" if v >= 0 else "#e74c3c"


def summary_row(pf: dict, perf: dict) -> html.Div:
    open_pnl = pf["unrealized_pnl"]
    real_pnl = pf["realized_pnl"]
    total    = pf["total_pnl"]
    invested = pf["total_invested"]
    ret_pct  = (total / invested * 100) if invested else 0

    cols = [
        card_metric("Portfolio Value",  f"${pf['total_value']:,.2f}", f"Invested ${invested:,.2f}"),
        card_metric("Total P&L",        f"${total:+,.2f}",            f"{ret_pct:+.2f}% total",  pnl_color(total)),
        card_metric("Unrealized P&L",   f"${open_pnl:+,.2f}",        "Open positions",           pnl_color(open_pnl)),
        card_metric("Realized P&L",     f"${real_pnl:+,.2f}",        "Closed trades",            pnl_color(real_pnl)),
        card_metric("Win Rate",         f"{perf.get('win_rate') or 0:.1f}%",
                    f"{perf.get('total_trades') or 0} closed trades"),
        card_metric("Avg Trade Return", f"{perf.get('avg_return') or 0:+.2f}%",
                    f"Sharpe≈{perf.get('sharpe') or 0:.2f}"),
    ]
    return dbc.Row([dbc.Col(c, width=2) for c in cols], className="g-2 mb-3")


TABLE_STYLE = dict(
    style_header={"backgroundColor": "#2c2c3e", "color": "white",
                  "fontWeight": "bold", "border": f"1px solid {BORDER}"},
    style_data={"backgroundColor": CARD, "color": "#ddd", "border": f"1px solid {BORDER}"},
    style_cell={"padding": "6px 10px", "fontSize": "12px", "fontFamily": "monospace"},
    style_table={"overflowX": "auto"},
    sort_action="native",
    page_size=20,
)


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

def create_app(debug: bool = False) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
        title="Trading Journal",
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div([
        dcc.Store(id="store-trades"),
        dcc.Store(id="store-selected-position"),

        # ── Navbar ───────────────────────────────
        dbc.Navbar(dbc.Container([
            html.Span("📒 Trading Journal", className="navbar-brand mb-0 h1",
                      style={"fontWeight": "bold", "fontSize": "1.3rem"}),
            dbc.Button("➕ Log Trade", id="btn-open-modal", color="success", size="sm"),
        ], fluid=True), color="dark", dark=True, sticky="top",
        style={"borderBottom": f"1px solid {BORDER}"}),

        # ── Trade Entry Modal ──────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Log a Trade")),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Symbol"),
                        dbc.Input(id="inp-symbol", placeholder="e.g. AAPL", type="text",
                                  style={"textTransform": "uppercase"}),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Side"),
                        dbc.Select(id="inp-side", options=[
                            {"label": "BUY  🟢", "value": "BUY"},
                            {"label": "SELL 🔴", "value": "SELL"},
                        ], value="BUY"),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Date"),
                        dbc.Input(id="inp-date", type="date",
                                  value=date.today().isoformat()),
                    ], md=4),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Entry Price ($)"),
                        dbc.Input(id="inp-price", type="number", min=0,
                                  placeholder="e.g. 185.50"),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Quantity"),
                        dbc.Input(id="inp-qty", type="number", min=0.001,
                                  placeholder="e.g. 10"),
                    ], md=6),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Strategy / Notes (optional)"),
                        dbc.Input(id="inp-notes", type="text",
                                  placeholder="e.g. RSI oversold bounce, TSMOM signal..."),
                    ]),
                ]),
                html.Div(id="modal-feedback", className="mt-2"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="btn-modal-cancel",
                           color="secondary", className="me-2"),
                dbc.Button("Log Trade", id="btn-modal-submit",
                           color="success"),
            ]),
        ], id="modal-trade", is_open=False, size="lg"),

        # ── Delete confirm modal ─────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Delete Trade")),
            dbc.ModalBody("Are you sure you want to delete this trade?"),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="btn-del-cancel", color="secondary", className="me-2"),
                dbc.Button("Delete", id="btn-del-confirm", color="danger"),
            ]),
        ], id="modal-delete", is_open=False),
        dcc.Store(id="store-delete-id"),

        # ── Main content ─────────────────────────
        dbc.Container([
            html.Div(id="summary-section", className="mt-3"),
            dbc.Tabs([

                # Portfolio tab
                dbc.Tab(label="💼 Portfolio", tab_id="tab-portfolio", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="graph-pnl"),  md=8),
                        dbc.Col(dcc.Graph(id="graph-alloc"), md=4),
                    ], className="mt-3"),
                    html.H6("Open Positions", className="mt-3",
                            style={"color": "#4fc3f7"}),
                    html.Div(id="table-open-positions"),
                ]),

                # Position detail tab
                dbc.Tab(label="🔍 Position Detail", tab_id="tab-detail", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Position:", className="mt-3"),
                            dcc.Dropdown(id="dd-position", placeholder="Choose a symbol...",
                                         style={"color": "#000"}),
                        ], md=4),
                    ]),
                    html.Div(id="detail-badges", className="mt-2 mb-2"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="graph-position-price"), md=8),
                        dbc.Col(dcc.Graph(id="graph-position-signals"), md=4),
                    ]),
                    html.Div(id="detail-signal-table"),
                ]),

                # Performance tab
                dbc.Tab(label="📈 Performance", tab_id="tab-perf", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="graph-closed-pnl"),
                        ], md=12),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col(html.Div(id="perf-stats-cards"), md=12),
                    ], className="mt-2"),
                ]),

                # Trade log tab
                dbc.Tab(label="📋 Trade Log", tab_id="tab-log", children=[
                    html.Div(className="mt-3", children=[
                        html.H6("All Trades", style={"color": "#4fc3f7"}),
                        html.Div(id="table-trade-log"),
                    ]),
                ]),

                # Closed trades tab
                dbc.Tab(label="✅ Closed Trades", tab_id="tab-closed", children=[
                    html.Div(className="mt-3", children=[
                        html.H6("Realized P&L — Closed Trade History",
                                style={"color": "#4fc3f7"}),
                        html.Div(id="table-closed-trades"),
                    ]),
                ]),

            ], id="main-tabs", active_tab="tab-portfolio"),
        ], fluid=True, style={"maxWidth": "1600px"}),

    ], style={"backgroundColor": DARK, "minHeight": "100vh",
              "color": "white", "fontFamily": "Inter, sans-serif"})

    # ─────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────

    # Initialize store from disk
    @app.callback(Output("store-trades", "data"), Input("store-trades", "id"))
    def init_trades(_):
        return load_trades()

    # Open/close modal
    @app.callback(
        Output("modal-trade", "is_open"),
        Output("modal-feedback", "children"),
        Input("btn-open-modal", "n_clicks"),
        Input("btn-modal-cancel", "n_clicks"),
        Input("btn-modal-submit", "n_clicks"),
        State("modal-trade", "is_open"),
        State("inp-symbol", "value"),
        State("inp-side",   "value"),
        State("inp-date",   "value"),
        State("inp-price",  "value"),
        State("inp-qty",    "value"),
        State("inp-notes",  "value"),
        State("store-trades", "data"),
        prevent_initial_call=True,
    )
    def toggle_modal(open_clicks, cancel_clicks, submit_clicks,
                     is_open, symbol, side, trade_date, price, qty, notes, trades):
        triggered = ctx.triggered_id
        if triggered == "btn-open-modal":
            return True, ""
        if triggered == "btn-modal-cancel":
            return False, ""
        if triggered == "btn-modal-submit":
            # Validate
            errors = []
            if not symbol: errors.append("Symbol required.")
            if not price:  errors.append("Price required.")
            if not qty:    errors.append("Quantity required.")
            if float(price or 0) <= 0: errors.append("Price must be > 0.")
            if float(qty   or 0) <= 0: errors.append("Quantity must be > 0.")
            if errors:
                return True, dbc.Alert(" ".join(errors), color="danger", className="mb-0")
            # Save
            trade = {
                "id":       next_id(trades or []),
                "symbol":   symbol.strip().upper(),
                "side":     side,
                "date":     trade_date,
                "price":    float(price),
                "quantity": float(qty),
                "notes":    notes or "",
                "logged_at": datetime.now().isoformat(),
            }
            trades = (trades or []) + [trade]
            save_trades(trades)
            # Clear price cache so next refresh gets live price
            _price_cache.pop(trade["symbol"], None)
            _history_cache.pop(trade["symbol"], None)
            return False, ""
        return is_open, ""

    # Re-read trades from disk whenever modal closes (to pick up saved trade)
    @app.callback(
        Output("store-trades", "data", allow_duplicate=True),
        Input("modal-trade", "is_open"),
        prevent_initial_call=True,
    )
    def reload_on_close(is_open):
        if not is_open:
            return load_trades()
        return dash.no_update

    # Delete modal
    @app.callback(
        Output("modal-delete", "is_open"),
        Output("store-delete-id", "data"),
        Input({"type": "btn-delete", "index": ALL}, "n_clicks"),
        Input("btn-del-cancel", "n_clicks"),
        Input("btn-del-confirm", "n_clicks"),
        State("store-delete-id", "data"),
        State("store-trades", "data"),
        prevent_initial_call=True,
    )
    def handle_delete(del_clicks, cancel, confirm, del_id, trades):
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "btn-delete":
            idx = triggered["index"]
            return True, idx
        if triggered == "btn-del-cancel":
            return False, None
        if triggered == "btn-del-confirm" and del_id is not None:
            updated = [t for t in (trades or []) if t.get("id") != del_id]
            save_trades(updated)
            return False, None
        return False, None

    @app.callback(
        Output("store-trades", "data", allow_duplicate=True),
        Input("modal-delete", "is_open"),
        prevent_initial_call=True,
    )
    def reload_after_delete(is_open):
        if not is_open:
            return load_trades()
        return dash.no_update

    # Master render callback
    @app.callback(
        Output("summary-section",       "children"),
        Output("graph-pnl",             "figure"),
        Output("graph-alloc",           "figure"),
        Output("table-open-positions",  "children"),
        Output("dd-position",           "options"),
        Output("graph-closed-pnl",      "figure"),
        Output("perf-stats-cards",      "children"),
        Output("table-closed-trades",   "children"),
        Output("table-trade-log",       "children"),
        Input("store-trades",           "data"),
    )
    def render_all(trades):
        trades = trades or []
        pf   = compute_portfolio(trades)
        perf = compute_performance(pf)

        summary = summary_row(pf, perf)

        pnl_fig   = build_pnl_chart(pf)
        alloc_fig = build_allocation_pie(pf)

        # Open positions table
        open_rows = list(pf["open"].values())
        if open_rows:
            df_open = pd.DataFrame(open_rows)[
                ["symbol","quantity","avg_cost","current_price",
                 "unrealized_pnl","return_pct","invested","hold_days"]
            ].rename(columns={"avg_cost":"avg cost","current_price":"price now",
                               "unrealized_pnl":"unreal P&L","return_pct":"ret %",
                               "hold_days":"days held"})
            df_open = df_open.round(4)
            pos_table = dash_table.DataTable(
                data=df_open.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_open.columns],
                style_data_conditional=[
                    {"if": {"filter_query": "{ret %} > 0", "column_id": "ret %"},
                     "color": "#27ae60"},
                    {"if": {"filter_query": "{ret %} < 0", "column_id": "ret %"},
                     "color": "#e74c3c"},
                    {"if": {"filter_query": "{unreal P&L} > 0", "column_id": "unreal P&L"},
                     "color": "#27ae60"},
                    {"if": {"filter_query": "{unreal P&L} < 0", "column_id": "unreal P&L"},
                     "color": "#e74c3c"},
                ],
                **TABLE_STYLE,
            )
        else:
            pos_table = dbc.Alert("No open positions. Log a BUY trade to get started.",
                                   color="secondary")

        # Dropdown options
        dd_opts = [{"label": sym, "value": sym} for sym in pf["open"]]

        # Closed P&L chart
        closed_fig = build_closed_pnl(pf["closed"])

        # Performance stat cards
        if perf["total_trades"]:
            best  = perf["best_trade"]
            worst = perf["worst_trade"]
            perf_cards = dbc.Row([
                dbc.Col(card_metric("Win Rate",       f"{perf['win_rate']}%",
                        f"{perf['total_trades']} trades")),
                dbc.Col(card_metric("Avg Return",     f"{perf['avg_return']:+.2f}%",
                        "per closed trade")),
                dbc.Col(card_metric("Trade Sharpe",   str(perf["sharpe"]),
                        "avg_ret / std")),
                dbc.Col(card_metric("Avg Hold",       f"{perf['avg_hold_days']} days")),
                dbc.Col(card_metric("Best Trade",
                        f"{best['symbol']} +{best['return_pct']:.1f}%",
                        f"${best['realized_pnl']:+.2f}",  "#27ae60")),
                dbc.Col(card_metric("Worst Trade",
                        f"{worst['symbol']} {worst['return_pct']:.1f}%",
                        f"${worst['realized_pnl']:+.2f}", "#e74c3c")),
            ], className="g-2")
        else:
            perf_cards = dbc.Alert("No closed trades yet. Close a position to see performance metrics.", color="secondary")

        # Closed trades table
        if pf["closed"]:
            df_c = pd.DataFrame(pf["closed"])
            df_c = df_c.round(4)
            closed_table = dash_table.DataTable(
                data=df_c.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_c.columns],
                style_data_conditional=[
                    {"if": {"filter_query": "{return_pct} > 0", "column_id": "return_pct"},
                     "color": "#27ae60"},
                    {"if": {"filter_query": "{return_pct} < 0", "column_id": "return_pct"},
                     "color": "#e74c3c"},
                ],
                **TABLE_STYLE,
            )
        else:
            closed_table = dbc.Alert("No closed trades yet.", color="secondary")

        # Trade log with delete buttons
        if trades:
            rows = []
            for t in sorted(trades, key=lambda x: x["date"], reverse=True):
                side_badge = dbc.Badge(
                    t["side"], color="success" if t["side"] == "BUY" else "danger",
                    className="me-1"
                )
                total = round(t["price"] * t["quantity"], 2)
                rows.append(html.Tr([
                    html.Td(t.get("id", "")),
                    html.Td(html.B(t["symbol"])),
                    html.Td(side_badge),
                    html.Td(t["date"]),
                    html.Td(f"${t['price']:.4f}"),
                    html.Td(t["quantity"]),
                    html.Td(f"${total:,.2f}"),
                    html.Td(t.get("notes", ""), style={"color": "#aaa", "fontSize": "11px"}),
                    html.Td(dbc.Button("🗑", id={"type": "btn-delete", "index": t.get("id")},
                                       color="danger", size="sm", outline=True)),
                ]))
            log_table = dbc.Table(
                [html.Thead(html.Tr([
                    html.Th("#"), html.Th("Symbol"), html.Th("Side"), html.Th("Date"),
                    html.Th("Price"), html.Th("Qty"), html.Th("Total"), html.Th("Notes"), html.Th("")
                ])),
                 html.Tbody(rows)],
                bordered=False, hover=True, responsive=True, size="sm",
                style={"color": "white"},
            )
        else:
            log_table = dbc.Alert("No trades logged yet. Click ➕ Log Trade to get started.",
                                   color="secondary")

        return (summary, pnl_fig, alloc_fig, pos_table,
                dd_opts, closed_fig, perf_cards, closed_table, log_table)

    # Position detail callback
    @app.callback(
        Output("graph-position-price",   "figure"),
        Output("graph-position-signals", "figure"),
        Output("detail-badges",          "children"),
        Output("detail-signal-table",    "children"),
        Input("dd-position",             "value"),
        State("store-trades",            "data"),
    )
    def update_detail(symbol, trades):
        if not symbol:
            return go.Figure(), go.Figure(), [], html.Div()

        pf  = compute_portfolio(trades or [])
        pos = pf["open"].get(symbol)
        if not pos:
            return go.Figure(), go.Figure(), [], html.Div()

        df   = get_history(symbol, days=365)
        sigs = run_signals(df)
        sig_label, sig_color = signal_summary(sigs)

        price_fig  = build_price_with_entry(
            symbol, pos["avg_cost"], pos["open_date"], pos["current_price"]
        )
        signal_fig = build_signal_bars(sigs, symbol)

        # Badges
        ret_c = "#27ae60" if pos["return_pct"] >= 0 else "#e74c3c"
        badges = [
            dbc.Badge(sig_label,  color="success" if "BUY" in sig_label else
                     ("danger" if "SELL" in sig_label else "secondary"),
                     className="me-2 fs-6"),
            dbc.Badge(f"Return: {pos['return_pct']:+.2f}%", className="me-2 fs-6",
                     style={"backgroundColor": ret_c}),
            dbc.Badge(f"P&L: ${pos['unrealized_pnl']:+.2f}", className="me-2 fs-6",
                     style={"backgroundColor": ret_c}),
            dbc.Badge(f"{pos['hold_days']} days held", color="info", className="me-2 fs-6"),
        ]

        # Signal detail table
        if sigs:
            rows = []
            for name, (vote, label) in sigs.items():
                icon  = "▲ BUY" if vote > 0 else ("▼ SELL" if vote < 0 else "— Neutral")
                color = "#27ae60" if vote > 0 else ("#e74c3c" if vote < 0 else "#777")
                rows.append(html.Tr([
                    html.Td(name, style={"fontWeight": "bold"}),
                    html.Td(icon, style={"color": color}),
                    html.Td(label, style={"color": "#ccc", "fontSize": "12px"}),
                ]))
            total_score = sum(v for v, _ in sigs.values())
            rows.append(html.Tr([
                html.Td(html.B("Total Score"), style={"borderTop": "1px solid #555"}),
                html.Td(html.B(f"{total_score:+d} / {len(sigs)}"),
                        style={"color": sig_color, "borderTop": "1px solid #555",
                               "fontWeight": "bold"}),
                html.Td(html.B(sig_label),
                        style={"color": sig_color, "borderTop": "1px solid #555"}),
            ]))
            sig_table = dbc.Table(
                [html.Thead(html.Tr([html.Th("Signal"), html.Th("Vote"), html.Th("Reason")])),
                 html.Tbody(rows)],
                bordered=False, size="sm", style={"color": "white"}, className="mt-2",
            )
        else:
            sig_table = dbc.Alert("Insufficient data for signals.", color="secondary")

        return price_fig, signal_fig, badges, sig_table

    return app


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Manual trading journal + exercise dashboard")
    parser.add_argument("--port",  type=int, default=8051, help="Port (default 8051)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Clear all trades")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(TRADES_FILE):
            os.remove(TRADES_FILE)
            print("  trades.json cleared.")

    app = create_app(debug=args.debug)

    trades = load_trades()
    print(f"\n{'═'*55}")
    print(f"  Trading Journal  —  {len(trades)} trades loaded")
    print(f"  Dashboard → http://localhost:{args.port}")
    print(f"  Ctrl+C to stop")
    print(f"{'═'*55}\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

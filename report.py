"""
report.py — Interactive Trading Dashboard
==========================================
Runs a Dash web app on http://localhost:8050

Features:
  • Live scan  — trigger a fresh screen.py scan from the browser
  • Load CSV   — auto-loads the latest saved screen_*.csv result
  • Overview   — summary cards + signal score distribution
  • Buy / Sell — ranked tables with sparklines and signal badges
  • Asset View — click any row → full price chart + signal breakdown
  • Heatmap    — full universe signal matrix
  • Raw Table  — filterable / sortable full results

Usage:
  python report.py                  # load latest CSV + serve on :8050
  python report.py --port 8080      # custom port
  python report.py --scan fast      # run fresh scan first, then serve
  python report.py --debug          # enable Dash debug mode
"""

import argparse
import glob
import os
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

SIGNAL_COLORS = {
    "BUY":       "#27ae60",
    "WEAK BUY":  "#2ecc71",
    "NEUTRAL":   "#95a5a6",
    "WEAK SELL": "#e67e22",
    "SELL":      "#e74c3c",
}

BADGE_COLORS = {
    "BUY":       "success",
    "WEAK BUY":  "info",
    "NEUTRAL":   "secondary",
    "WEAK SELL": "warning",
    "SELL":      "danger",
}


def load_latest_csv() -> pd.DataFrame:
    """Find and load the most recent screen_*.csv file."""
    files = sorted(glob.glob("screen_*.csv"))
    if not files:
        files = sorted(glob.glob("backtest_results_*.csv"))
    if not files:
        return pd.DataFrame()
    latest = files[-1]
    print(f"  Loading {latest}")
    df = pd.read_csv(latest)
    # Normalise column names across screen.py and backtest.py outputs
    if "signal" not in df.columns and "bt_sharpe" in df.columns:
        # backtest output — synthesise signal column
        df["signal"] = df.apply(
            lambda r: "BUY" if r.get("buy_votes", 0) > r.get("sell_votes", 0) else
                      ("SELL" if r.get("sell_votes", 0) > r.get("buy_votes", 0) else "NEUTRAL"),
            axis=1
        )
        df["score"] = df.get("buy_votes", 0) - df.get("sell_votes", 0)
    return df


_name_cache: dict = {}

# ── Known ETF prefixes / crypto patterns ─────────────────────────────────
_ETF_SUFFIXES = {"SPY","QQQ","IWM","DIA","GLD","SLV","TLT","HYG","LQD","VXX",
                 "ARKK","ARKG","ARKF","ARKQ","ARKW","XLK","XLF","XLE","XLV",
                 "XLI","XLP","XLU","XLY","XLB","XLRE","IAU","PPLT","PDBC",
                 "VTI","VOO","VEA","VWO","IEFA","EEM","EFA","AGG","BND",
                 "BNDX","EMB","SCHD","JEPI","QYLD","XYLD","RSP","MDY","IJR",
                 "IVV","ITOT","ESGU","SCHF","SPDW","SPEM","SPIB","SPSB",
                 "GOVT","SHY","IEF","VGIT","VCIT","VCSH","VMBS","MBB","MUB"}

_CRYPTO_SUFFIXES = {"-USD","-BTC","-ETH","USDT","BUSD","BTC","ETH","BNB","SOL","ADA",
                    "DOGE","DOT","AVAX","MATIC","LINK","UNI","ATOM","XRP","LTC","XLM"}


def classify_asset(symbol: str) -> dict:
    """Return {type, exchange} for a ticker symbol, derived from its format."""
    s = symbol.upper()

    # Crypto: ends with -USD, -BTC, -ETH, or known suffix
    if any(s.endswith(sfx) for sfx in ["-USD","-BTC","-ETH","-USDT"]):
        return {"type": "Crypto", "exchange": "Crypto"}
    if s in {"BTC","ETH","BNB","SOL","ADA","DOGE","XRP","DOT","AVAX","MATIC","LINK"}:
        return {"type": "Crypto", "exchange": "Crypto"}

    # International exchanges by suffix
    if "." in s:
        sfx = s.split(".")[-1]
        exch_map = {
            "SW": ("Equity", "SIX (Switzerland)"),
            "PA": ("Equity", "Euronext Paris"),
            "L":  ("Equity", "London Stock Exchange"),
            "DE": ("Equity", "XETRA (Germany)"),
            "AS": ("Equity", "Euronext Amsterdam"),
            "MC": ("Equity", "Bolsa Madrid"),
            "MI": ("Equity", "Borsa Milano"),
            "HK": ("Equity", "Hong Kong Stock Exchange"),
            "T":  ("Equity", "Tokyo Stock Exchange"),
            "SS": ("Equity", "Shanghai Stock Exchange"),
            "SZ": ("Equity", "Shenzhen Stock Exchange"),
            "AX": ("Equity", "ASX (Australia)"),
            "TO": ("Equity", "Toronto Stock Exchange"),
            "BR": ("Equity", "Euronext Brussels"),
            "LS": ("Equity", "Euronext Lisbon"),
            "F":  ("Equity", "Frankfurt Stock Exchange"),
            "VI": ("Equity", "Vienna Stock Exchange"),
            "ST": ("Equity", "Nasdaq Stockholm"),
            "CO": ("Equity", "Nasdaq Copenhagen"),
            "OL": ("Equity", "Oslo Stock Exchange"),
            "HE": ("Equity", "Nasdaq Helsinki"),
        }
        if sfx in exch_map:
            return {"type": exch_map[sfx][0], "exchange": exch_map[sfx][1]}

    # ETFs
    if s in _ETF_SUFFIXES or (len(s) >= 3 and s.endswith(("ETF","ETF2"))):
        return {"type": "ETF", "exchange": "US Markets"}

    # Futures / bonds / macro (common patterns)
    if s.startswith("^") or s in {"GC=F","SI=F","CL=F","NG=F","ZC=F","ZW=F","ZS=F",
                                   "ES=F","NQ=F","RTY=F","YM=F","GE=F","TN=F"}:
        return {"type": "Futures/Index", "exchange": "US Markets"}

    # Default: US Equity
    return {"type": "Equity", "exchange": "US Markets (NYSE/NASDAQ)"}


def get_asset_name(symbol: str) -> str:
    """Return full company name for a ticker symbol, cached."""
    if symbol in _name_cache:
        return _name_cache[symbol]
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        name = info.get("longName") or info.get("shortName") or symbol
        _name_cache[symbol] = name
    except Exception:
        name = symbol
        _name_cache[symbol] = symbol
    return name


def fetch_price_history(symbol: str, days: int = 365) -> pd.DataFrame:
    """Download OHLCV for a single symbol."""
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
        df.index.name = "date"
        return df
    except Exception:
        return pd.DataFrame()


def signal_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("sig_")]


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────

def build_score_distribution(df: pd.DataFrame) -> go.Figure:
    if df.empty or "score" not in df.columns:
        return go.Figure()
    counts = df["score"].value_counts().sort_index()
    colors = ["#e74c3c" if x < 0 else ("#27ae60" if x > 0 else "#95a5a6")
              for x in counts.index]
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors,
        hovertemplate="Score %{x}: %{y} assets<extra></extra>",
    ))
    fig.add_vline(x=3,  line_dash="dot", line_color="#27ae60", annotation_text="BUY  ≥+3")
    fig.add_vline(x=-3, line_dash="dot", line_color="#e74c3c", annotation_text="SELL ≤-3")
    fig.update_layout(
        title="Signal Score Distribution", xaxis_title="Score",
        yaxis_title="Assets", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font_color="white", margin=dict(t=40, b=30, l=40, r=20),
        xaxis=dict(tickmode="linear", gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


def build_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                  x_label: str, y_label: str, title: str) -> go.Figure:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return go.Figure()
    fig = go.Figure()
    for sig, color in SIGNAL_COLORS.items():
        sub = df[df["signal"] == sig]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col], mode="markers",
            name=sig, marker=dict(color=color, size=7, opacity=0.75),
            text=sub["symbol"],
            hovertemplate="<b>%{text}</b><br>" + x_label + ": %{x:.2f}<br>" +
                          y_label + ": %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title=title, xaxis_title=x_label, yaxis_title=y_label,
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="white",
        margin=dict(t=40, b=40, l=50, r=20),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        legend=dict(bgcolor="#1e1e2e"),
    )
    return fig


def build_price_chart(symbol: str, df: pd.DataFrame) -> go.Figure:
    """Candlestick + volume + SMA50/200 + Bollinger."""
    if df.empty:
        return go.Figure().update_layout(title=f"No data for {symbol}")

    c = df["close"].astype(float)
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    mid    = c.rolling(20).mean()
    std_   = c.rolling(20).std()
    bb_up  = mid + 2 * std_
    bb_lo  = mid - 2 * std_

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="OHLC", increasing_line_color="#27ae60",
        decreasing_line_color="#e74c3c",
    ), row=1, col=1)

    # SMA lines
    fig.add_trace(go.Scatter(x=df.index, y=sma50,  mode="lines",
        line=dict(color="#f39c12", width=1.5, dash="dot"), name="SMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma200, mode="lines",
        line=dict(color="#3498db", width=1.5, dash="dot"), name="SMA200"), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=bb_up, mode="lines",
        line=dict(color="rgba(150,150,255,0.4)", width=1), name="BB Upper",
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lo, mode="lines",
        line=dict(color="rgba(150,150,255,0.4)", width=1), name="BB Lower",
        fill="tonexty", fillcolor="rgba(150,150,255,0.05)",
        showlegend=False), row=1, col=1)

    # Volume
    vol_colors = ["#27ae60" if c >= o else "#e74c3c"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"],
        name="Volume", marker_color=vol_colors, opacity=0.6), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} — Price Chart",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="white",
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#1e1e2e", font_size=11),
        margin=dict(t=50, b=20, l=50, r=20),
        height=500,
    )
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    return fig


def build_signal_radar(row: pd.Series) -> go.Figure:
    """Radar chart of individual signal votes for one asset."""
    sig_map = {c.replace("sig_", ""): v for c, v in row.items()
               if c.startswith("sig_")}
    if not sig_map:
        return go.Figure()
    cats   = list(sig_map.keys())
    vals   = [float(sig_map[k]) for k in cats]
    colors = ["#27ae60" if v > 0 else ("#e74c3c" if v < 0 else "#95a5a6") for v in vals]
    fig = go.Figure(go.Bar(
        x=cats, y=vals,
        marker_color=colors,
        hovertemplate="%{x}: %{y:+d}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    fig.update_layout(
        title=f"{row.get('symbol','?')} — Signal Breakdown",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="white",
        yaxis=dict(tickvals=[-1, 0, 1], range=[-1.4, 1.4], gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        margin=dict(t=40, b=30, l=40, r=20),
        height=280,
    )
    return fig


def build_heatmap(df: pd.DataFrame, max_rows: int = 60) -> go.Figure:
    scols = signal_cols(df)
    if df.empty or not scols:
        return go.Figure()
    # Show most actionable assets
    sub = df[df["signal"].isin(["BUY","SELL","WEAK BUY","WEAK SELL"])]\
            .sort_values("score", ascending=False).head(max_rows)
    if sub.empty:
        sub = df.sort_values("score", ascending=False).head(max_rows)
    heat = sub[scols].astype(float)
    syms = sub["symbol"].tolist()
    labels = [c.replace("sig_", "") for c in scols]
    text_matrix = [[{1: "▲", -1: "▼", 0: "─"}.get(int(v), "") for v in row]
                   for row in heat.values]
    fig = go.Figure(go.Heatmap(
        z=heat.values.T, x=syms, y=labels,
        colorscale=[[0,"#e74c3c"], [0.5,"#2c2c3e"], [1,"#27ae60"]],
        zmin=-1, zmax=1,
        text=[[row[i] for row in text_matrix] for i in range(len(labels))],
        texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="<b>%{x}</b> — %{y}: %{z:+d}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title="Signal", tickvals=[-1,0,1],
            ticktext=["SELL","NEUTRAL","BUY"],
            bgcolor="#1e1e2e", tickfont=dict(color="white"),
        ),
    ))
    fig.update_layout(
        title="Signal Heatmap — All Actionable Assets",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e", font_color="white",
        margin=dict(t=50, b=100, l=80, r=20),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        height=max(350, len(labels) * 36),
    )
    return fig


# ─────────────────────────────────────────────
# TABLE BUILDER
# ─────────────────────────────────────────────

def make_table_data(df: pd.DataFrame, signal_filter: Optional[list] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    sub = df.copy()
    if signal_filter:
        sub = sub[sub["signal"].isin(signal_filter)]

    sub = sub.copy()
    # Asset type + exchange classification (free, no network)
    cls = sub["symbol"].apply(classify_asset)
    sub["type"]     = cls.apply(lambda x: x["type"])
    sub["exchange"] = cls.apply(lambda x: x["exchange"])
    # Company name (network, top-50 only)
    top_syms = sub["symbol"].head(50).tolist() if "symbol" in sub.columns else []
    name_map = {s: get_asset_name(s) for s in top_syms}
    sub["name"] = sub["symbol"].map(name_map).fillna(sub["symbol"])

    display_cols = ["symbol", "name", "type", "exchange", "price", "signal", "score", "confidence",
                    "ret_1d", "ret_5d", "ret_1mo", "ret_3mo", "vol_ann", "reason"]
    # Fallback columns for backtest output
    if "ret_1d" not in sub.columns:
        for c in ["ret_1d","ret_5d","ret_1mo","ret_3mo","vol_ann"]:
            sub[c] = sub.get(c, np.nan)
    if "reason" not in sub.columns:
        sub["reason"] = sub.get("best_strategy", "")
    if "confidence" not in sub.columns:
        sub["confidence"] = (sub.get("score", sub.get("buy_votes", 0) - sub.get("sell_votes", 0)).abs() / 8 * 100).round(1)

    existing = [c for c in display_cols if c in sub.columns]
    out = sub[existing].copy()

    # Round numeric columns
    for col in ["price", "score", "confidence", "ret_1d", "ret_5d", "ret_1mo", "ret_3mo", "vol_ann"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    return out.sort_values("score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# LAYOUT BUILDERS
# ─────────────────────────────────────────────

def summary_cards(df: pd.DataFrame):
    if df.empty:
        return dbc.Row([dbc.Col(dbc.Alert("No data loaded. Run a scan first.", color="warning"))])

    counts = df["signal"].value_counts() if "signal" in df.columns else {}
    total  = len(df)

    def card(title, value, color, icon):
        return dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div(icon, style={"fontSize": "2rem", "lineHeight": "1"}),
                html.H3(str(value), className="mb-0 mt-1",
                        style={"color": color, "fontWeight": "bold"}),
                html.P(title, className="mb-0 text-muted", style={"fontSize": "0.85rem"}),
            ])
        ], style={"background": "#1e1e2e", "border": f"1px solid {color}",
                  "borderRadius": "10px", "textAlign": "center"}),
        width=2)

    return dbc.Row([
        card("Total Scanned", total,                               "#ffffff", "📊"),
        card("Strong BUY",    counts.get("BUY",      0),          "#27ae60", "🟢"),
        card("Weak BUY",      counts.get("WEAK BUY", 0),          "#2ecc71", "🟩"),
        card("Neutral",       counts.get("NEUTRAL",  0),          "#95a5a6", "⬜"),
        card("Weak SELL",     counts.get("WEAK SELL",0),          "#e67e22", "🟧"),
        card("Strong SELL",   counts.get("SELL",     0),          "#e74c3c", "🔴"),
    ], className="g-2")


def signal_badge(signal: str) -> dbc.Badge:
    return dbc.Badge(signal, color=BADGE_COLORS.get(signal, "secondary"),
                     className="ms-1", pill=True)


TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_header": {"backgroundColor": "#2c2c3e", "color": "white",
                     "fontWeight": "bold", "border": "1px solid #444"},
    "style_data":   {"backgroundColor": "#1e1e2e", "color": "#ddd",
                     "border": "1px solid #333"},
    "style_data_conditional": [
        {"if": {"filter_query": '{signal} = "BUY"',    "column_id": "signal"},
         "color": "#27ae60", "fontWeight": "bold"},
        {"if": {"filter_query": '{signal} = "WEAK BUY"', "column_id": "signal"},
         "color": "#2ecc71"},
        {"if": {"filter_query": '{signal} = "SELL"',   "column_id": "signal"},
         "color": "#e74c3c", "fontWeight": "bold"},
        {"if": {"filter_query": '{signal} = "WEAK SELL"', "column_id": "signal"},
         "color": "#e67e22"},
        {"if": {"filter_query": "{score} >= 3", "column_id": "score"},
         "backgroundColor": "rgba(39,174,96,0.15)"},
        {"if": {"filter_query": "{score} <= -3", "column_id": "score"},
         "backgroundColor": "rgba(231,76,60,0.15)"},
    ],
    "style_cell": {"padding": "6px 12px", "fontSize": "13px",
                   "fontFamily": "monospace"},
    "page_size": 25,
    "sort_action": "native",
    "filter_action": "native",
    "row_selectable": "single",
}


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

def create_app(initial_df: pd.DataFrame, debug: bool = False) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
        title="Trading Dashboard",
        suppress_callback_exceptions=True,
    )

    # ── Store for shared data ────────────────
    app.layout = html.Div([
        dcc.Store(id="store-data", data=initial_df.to_dict("records") if not initial_df.empty else []),
        dcc.Store(id="store-selected-symbol", data=None),
        dcc.Interval(id="interval-clock", interval=60_000, n_intervals=0),

        # ── Navbar ───────────────────────────
        dbc.Navbar(
            dbc.Container([
                html.Span("📈 Trading Dashboard", className="navbar-brand mb-0 h1",
                          style={"fontWeight": "bold", "fontSize": "1.3rem"}),
                html.Div([
                    html.Span(id="last-updated",
                              style={"color": "#aaa", "marginRight": "20px", "fontSize": "0.85rem"}),
                    dbc.Button("🔄 Refresh Scan", id="btn-scan", color="success", size="sm",
                               className="me-2"),
                    dbc.Select(
                        id="select-universe",
                        options=[
                            {"label": "Fast (~250 assets)", "value": "fast"},
                            {"label": "S&P 500", "value": "sp500"},
                            {"label": "NASDAQ 100", "value": "nasdaq100"},
                            {"label": "ETFs", "value": "etf"},
                            {"label": "Crypto", "value": "crypto"},
                            {"label": "Full (~900 assets)", "value": "full"},
                        ],
                        value="fast", size="sm",
                        style={"width": "200px", "display": "inline-block"},
                    ),
                ], className="d-flex align-items-center"),
            ], fluid=True),
            color="dark", dark=True, sticky="top",
            style={"borderBottom": "1px solid #333"},
        ),

        # ── Scan progress toast ───────────────
        dbc.Toast(
            id="toast-scan",
            header="Scan Running...",
            is_open=False,
            dismissable=True,
            duration=4000,
            style={"position": "fixed", "top": 80, "right": 20, "zIndex": 9999},
            color="info",
        ),

        # ── Main content ─────────────────────
        dbc.Container([

            # Summary cards
            html.Div(id="summary-cards", className="my-3"),

            # ── Filter bar ───────────────────────────────────────────────
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Asset Type", className="text-muted",
                                   style={"fontSize": "0.8rem", "marginBottom": "2px"}),
                        dcc.Dropdown(
                            id="filter-type",
                            options=[
                                {"label": "All Types",           "value": "all"},
                                {"label": "Equity",              "value": "Equity"},
                                {"label": "ETF",                 "value": "ETF"},
                                {"label": "Crypto",              "value": "Crypto"},
                                {"label": "Futures/Index",       "value": "Futures/Index"},
                            ],
                            value="all", clearable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], md=2),
                    dbc.Col([
                        html.Label("Exchange / Region", className="text-muted",
                                   style={"fontSize": "0.8rem", "marginBottom": "2px"}),
                        dcc.Dropdown(
                            id="filter-exchange",
                            options=[
                                {"label": "All Exchanges",               "value": "all"},
                                {"label": "US Markets (NYSE/NASDAQ)",    "value": "US Markets (NYSE/NASDAQ)"},
                                {"label": "US Markets (ETFs)",           "value": "US Markets"},
                                {"label": "Crypto",                      "value": "Crypto"},
                                {"label": "SIX (Switzerland)",           "value": "SIX (Switzerland)"},
                                {"label": "London Stock Exchange",       "value": "London Stock Exchange"},
                                {"label": "XETRA (Germany)",             "value": "XETRA (Germany)"},
                                {"label": "Euronext Paris",              "value": "Euronext Paris"},
                                {"label": "Euronext Amsterdam",          "value": "Euronext Amsterdam"},
                                {"label": "Tokyo Stock Exchange",        "value": "Tokyo Stock Exchange"},
                                {"label": "Hong Kong Stock Exchange",    "value": "Hong Kong Stock Exchange"},
                                {"label": "ASX (Australia)",             "value": "ASX (Australia)"},
                                {"label": "Toronto Stock Exchange",      "value": "Toronto Stock Exchange"},
                                {"label": "Shanghai/Shenzhen",           "value": "Shanghai"},
                            ],
                            value="all", clearable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Signal Filter", className="text-muted",
                                   style={"fontSize": "0.8rem", "marginBottom": "2px"}),
                        dcc.Dropdown(
                            id="filter-signal",
                            options=[
                                {"label": "All Signals",   "value": "all"},
                                {"label": "BUY only",      "value": "BUY"},
                                {"label": "WEAK BUY",      "value": "WEAK BUY"},
                                {"label": "NEUTRAL",       "value": "NEUTRAL"},
                                {"label": "WEAK SELL",     "value": "WEAK SELL"},
                                {"label": "SELL only",     "value": "SELL"},
                                {"label": "Any BUY",       "value": "any_buy"},
                                {"label": "Any SELL",      "value": "any_sell"},
                            ],
                            value="all", clearable=False,
                            style={"fontSize": "13px"},
                        ),
                    ], md=2),
                    dbc.Col([
                        html.Label("Min Score", className="text-muted",
                                   style={"fontSize": "0.8rem", "marginBottom": "2px"}),
                        dcc.Slider(
                            id="filter-score",
                            min=-8, max=8, step=1, value=-8,
                            marks={i: str(i) for i in range(-8, 9, 2)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Max Vol %", className="text-muted",
                                   style={"fontSize": "0.8rem", "marginBottom": "2px"}),
                        dcc.Slider(
                            id="filter-vol",
                            min=0, max=200, step=10, value=200,
                            marks={0:"0", 30:"30", 60:"60", 100:"100", 200:"200"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ], md=2),
                ], className="g-2"),
                dbc.Row([
                    dbc.Col(html.Small(id="filter-count",
                                       style={"color": "#aaa", "marginTop": "6px"}), md=12),
                ]),
            ]), style={"background": "#1e1e2e", "border": "1px solid #333",
                       "borderRadius": "8px", "marginBottom": "16px"}),

            # Filtered store
            dcc.Store(id="store-filtered"),

            # Tabs
            dbc.Tabs([

                # ── Overview ─────────────────
                dbc.Tab(label="📊 Overview", tab_id="tab-overview", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="graph-distribution"), md=5),
                        dbc.Col(dcc.Graph(id="graph-scatter-ret"), md=7),
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="graph-scatter-vol"), md=6),
                        dbc.Col(dcc.Graph(id="graph-pie"), md=6),
                    ], className="mt-2"),
                ]),

                # ── Buy Signals ───────────────
                dbc.Tab(label="🟢 Buy Signals", tab_id="tab-buy", children=[
                    html.Div(className="mt-3", children=[
                        dbc.Row([
                            dbc.Col([
                                html.H6("Strong BUY (score ≥ +3)",
                                        style={"color": "#27ae60"}),
                                html.Div(id="table-buy-strong"),
                            ], md=12),
                        ]),
                        html.Hr(style={"borderColor": "#333"}),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Weak BUY (score = +2)",
                                        style={"color": "#2ecc71"}),
                                html.Div(id="table-buy-weak"),
                            ], md=12),
                        ]),
                    ]),
                ]),

                # ── Sell Signals ──────────────
                dbc.Tab(label="🔴 Sell Signals", tab_id="tab-sell", children=[
                    html.Div(className="mt-3", children=[
                        dbc.Row([
                            dbc.Col([
                                html.H6("Strong SELL (score ≤ -3)",
                                        style={"color": "#e74c3c"}),
                                html.Div(id="table-sell-strong"),
                            ], md=12),
                        ]),
                        html.Hr(style={"borderColor": "#333"}),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Weak SELL (score = -2)",
                                        style={"color": "#e67e22"}),
                                html.Div(id="table-sell-weak"),
                            ], md=12),
                        ]),
                    ]),
                ]),

                # ── Asset Detail ──────────────
                dbc.Tab(label="🔍 Asset Detail", tab_id="tab-detail", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Symbol"),
                                dbc.Input(id="input-symbol", value="AAPL",
                                          placeholder="e.g. AAPL, BTC-USD"),
                                dbc.Button("Load", id="btn-load-symbol",
                                           color="primary", n_clicks=0),
                            ], className="mt-3 mb-2", style={"maxWidth": "350px"}),
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="detail-signal-badges", className="mb-2"),
                            dcc.Graph(id="graph-price"),
                        ], md=8),
                        dbc.Col([
                            dcc.Graph(id="graph-signal-radar"),
                            html.Div(id="detail-stats", className="mt-2"),
                        ], md=4),
                    ]),
                ]),

                # ── Heatmap ───────────────────
                dbc.Tab(label="🗺 Heatmap", tab_id="tab-heatmap", children=[
                    dbc.Row([
                        dbc.Col([
                            html.P("Signal matrix for all actionable assets (BUY/SELL).",
                                   className="text-muted mt-3 mb-1",
                                   style={"fontSize": "0.85rem"}),
                            dcc.Graph(id="graph-heatmap"),
                        ]),
                    ]),
                ]),

                # ── Raw Data ──────────────────
                dbc.Tab(label="📋 Raw Data", tab_id="tab-raw", children=[
                    html.Div(id="table-raw", className="mt-3"),
                ]),

            ], id="tabs", active_tab="tab-overview"),

        ], fluid=True, style={"maxWidth": "1600px"}),

    ], style={"backgroundColor": "#12121f", "minHeight": "100vh",
              "color": "white", "fontFamily": "Inter, sans-serif"})

    # ─────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────

    @app.callback(
        Output("store-data", "data"),
        Output("toast-scan", "is_open"),
        Output("toast-scan", "children"),
        Output("toast-scan", "color"),
        Input("btn-scan", "n_clicks"),
        State("select-universe", "value"),
        prevent_initial_call=True,
    )
    def run_scan(n_clicks, universe):
        if not n_clicks:
            return dash.no_update, False, "", "info"
        try:
            result = subprocess.run(
                [sys.executable, "screen.py",
                 "--universe", universe, "--no-chart", "--days", "300"],
                capture_output=True, text=True, timeout=600
            )
            df = load_latest_csv()
            if df.empty:
                return [], True, "Scan finished but no data found.", "warning"
            return df.to_dict("records"), True, f"Scan complete — {len(df)} assets screened.", "success"
        except subprocess.TimeoutExpired:
            return dash.no_update, True, "Scan timed out.", "danger"
        except Exception as e:
            return dash.no_update, True, f"Error: {e}", "danger"

    @app.callback(
        Output("last-updated", "children"),
        Input("interval-clock", "n_intervals"),
        Input("store-data", "data"),
    )
    def update_clock(_, data):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n  = len(data) if data else 0
        return f"Last updated: {ts}  |  {n} assets"

    def _df_from_store(data):
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    # ── Filter callback: raw data → filtered store ────────────────────────
    @app.callback(
        Output("store-filtered", "data"),
        Output("filter-count",   "children"),
        Input("store-data",      "data"),
        Input("filter-type",     "value"),
        Input("filter-exchange", "value"),
        Input("filter-signal",   "value"),
        Input("filter-score",    "value"),
        Input("filter-vol",      "value"),
    )
    def apply_filters(data, ftype, fexchange, fsignal, fmin_score, fmax_vol):
        df = _df_from_store(data)
        if df.empty:
            return [], "No data loaded."

        # Classify each symbol
        cls = df["symbol"].apply(classify_asset)
        df  = df.copy()
        df["_type"]     = cls.apply(lambda x: x["type"])
        df["_exchange"] = cls.apply(lambda x: x["exchange"])

        # Apply type filter
        if ftype and ftype != "all":
            df = df[df["_type"] == ftype]

        # Apply exchange filter (partial match for Shanghai/Shenzhen)
        if fexchange and fexchange != "all":
            if fexchange == "Shanghai":
                df = df[df["_exchange"].isin(["Shanghai Stock Exchange", "Shenzhen Stock Exchange"])]
            else:
                df = df[df["_exchange"] == fexchange]

        # Apply signal filter
        if fsignal and fsignal != "all":
            if fsignal == "any_buy":
                df = df[df["signal"].isin(["BUY", "WEAK BUY"])]
            elif fsignal == "any_sell":
                df = df[df["signal"].isin(["SELL", "WEAK SELL"])]
            else:
                df = df[df["signal"] == fsignal]

        # Score filter
        if "score" in df.columns and fmin_score is not None:
            df = df[pd.to_numeric(df["score"], errors="coerce").fillna(0) >= fmin_score]

        # Volatility filter
        if "vol_ann" in df.columns and fmax_vol is not None and fmax_vol < 200:
            df = df[pd.to_numeric(df["vol_ann"], errors="coerce").fillna(999) <= fmax_vol]

        n_total = len(_df_from_store(data))
        label   = f"Showing {len(df)} of {n_total} assets after filters"
        return df.drop(columns=["_type","_exchange"]).to_dict("records"), label

    @app.callback(
        Output("summary-cards",      "children"),
        Output("graph-distribution", "figure"),
        Output("graph-scatter-ret",  "figure"),
        Output("graph-scatter-vol",  "figure"),
        Output("graph-pie",          "figure"),
        Output("graph-heatmap",      "figure"),
        Input("store-filtered",      "data"),
    )
    def update_overview(data):
        df = _df_from_store(data)

        cards = summary_cards(df)
        dist  = build_score_distribution(df)

        x_col = "ret_1mo" if "ret_1mo" in df.columns else "bt_return_pct"
        v_col = "vol_ann"  if "vol_ann"  in df.columns else "vol_20d_pct"

        scatter_ret = build_scatter(
            df, x_col, "score",
            "1-Month Return (%)", "Signal Score", "Score vs 1-Month Return"
        ) if not df.empty else go.Figure()

        scatter_vol = build_scatter(
            df, v_col, "score",
            "Annualised Volatility (%)", "Signal Score", "Score vs Volatility"
        ) if not df.empty else go.Figure()

        # Pie chart
        if not df.empty and "signal" in df.columns:
            counts = df["signal"].value_counts()
            pie = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                marker_colors=[SIGNAL_COLORS.get(s, "#aaa") for s in counts.index],
                hole=0.4, textinfo="label+percent",
            ))
            pie.update_layout(
                title="Signal Distribution", paper_bgcolor="#1e1e2e",
                font_color="white", margin=dict(t=40, b=20),
                showlegend=False,
            )
        else:
            pie = go.Figure()

        heatmap = build_heatmap(df)

        return cards, dist, scatter_ret, scatter_vol, pie, heatmap

    def _make_dash_table(df_data: pd.DataFrame, table_id: str) -> dash_table.DataTable:
        if df_data.empty:
            return html.P("No data.", style={"color": "#aaa"})
        records = df_data.to_dict("records")
        cols    = [{"name": c, "id": c} for c in df_data.columns]
        return dash_table.DataTable(
            id=table_id, data=records, columns=cols, **TABLE_STYLE,
        )

    @app.callback(
        Output("table-buy-strong", "children"),
        Output("table-buy-weak",   "children"),
        Output("table-sell-strong","children"),
        Output("table-sell-weak",  "children"),
        Output("table-raw",        "children"),
        Input("store-filtered", "data"),
    )
    def update_tables(data):
        df = _df_from_store(data)
        buy_s  = _make_dash_table(make_table_data(df, ["BUY"]),       "tbl-buy-s")
        buy_w  = _make_dash_table(make_table_data(df, ["WEAK BUY"]), "tbl-buy-w")
        sell_s = _make_dash_table(make_table_data(df, ["SELL"]),      "tbl-sell-s")
        sell_w = _make_dash_table(make_table_data(df, ["WEAK SELL"]),"tbl-sell-w")
        raw    = _make_dash_table(make_table_data(df),                "tbl-raw")
        return buy_s, buy_w, sell_s, sell_w, raw

    @app.callback(
        Output("graph-price",          "figure"),
        Output("graph-signal-radar",   "figure"),
        Output("detail-signal-badges", "children"),
        Output("detail-stats",         "children"),
        Input("btn-load-symbol",       "n_clicks"),
        State("input-symbol",          "value"),
        State("store-filtered",        "data"),
        prevent_initial_call=False,
    )
    def update_detail(n_clicks, symbol, data):
        symbol = (symbol or "AAPL").strip().upper()
        df_hist = fetch_price_history(symbol, days=365)
        price_fig = build_price_chart(symbol, df_hist)

        df = _df_from_store(data)
        row = df[df["symbol"] == symbol].iloc[0] if not df.empty and symbol in df["symbol"].values else pd.Series()

        radar_fig = build_signal_radar(row) if not row.empty else go.Figure()

        # Badges
        badges = []
        if not row.empty:
            sig = row.get("signal", "?")
            badges = [
                dbc.Badge(f"Signal: {sig}", color=BADGE_COLORS.get(sig, "secondary"),
                          className="me-2 fs-6"),
                dbc.Badge(f"Score: {row.get('score', '?'):+}", color="light",
                          text_color="dark", className="me-2 fs-6"),
                dbc.Badge(f"Confidence: {row.get('confidence', '?'):.0f}%",
                          color="info", className="me-2 fs-6"),
            ]

        # Stats card
        stats = html.Div()
        if not row.empty:
            stats = dbc.Card([dbc.CardBody([
                html.H6(f"{symbol} Statistics", className="card-title"),
                dbc.Table([html.Tbody([
                    html.Tr([html.Td("Price"),   html.Td(f"${row.get('price','?'):.4f}")]),
                    html.Tr([html.Td("1d Ret"),  html.Td(f"{row.get('ret_1d','?'):+.2f}%")]),
                    html.Tr([html.Td("5d Ret"),  html.Td(f"{row.get('ret_5d','?'):+.2f}%")]),
                    html.Tr([html.Td("1mo Ret"), html.Td(f"{row.get('ret_1mo','?'):+.2f}%")]),
                    html.Tr([html.Td("Vol"),     html.Td(f"{row.get('vol_ann','?'):.1f}%")]),
                    html.Tr([html.Td("Reason"),  html.Td(str(row.get('reason',''))[:60])]),
                ])], bordered=False, size="sm",
                style={"color": "white", "fontSize": "12px"}),
            ])], style={"background": "#1e1e2e", "border": "1px solid #333"})

        return price_fig, radar_fig, badges, stats

    return app


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive trading dashboard")
    parser.add_argument("--port",     type=int, default=8050, help="Local port (default 8050)")
    parser.add_argument("--scan",     metavar="UNIVERSE",     help="Run a fresh scan before launching")
    parser.add_argument("--debug",    action="store_true",    help="Enable Dash debug mode")
    parser.add_argument("--password", type=str, default=None,
                        help="Login password (or set DASHBOARD_PASSWORD env var)")
    args = parser.parse_args()

    # Optionally run a fresh scan first
    if args.scan:
        print(f"\nRunning fresh scan (universe={args.scan})...")
        subprocess.run([
            sys.executable, "screen.py",
            "--universe", args.scan,
            "--no-chart", "--days", "300"
        ])

    # Load data
    print("\nLoading screening data...")
    df = load_latest_csv()
    if df.empty:
        print("  No CSV found — launch the dashboard anyway (use the Refresh Scan button).")
    else:
        print(f"  Loaded {len(df)} assets.")

    app = create_app(df, debug=args.debug)

    from auth import add_auth
    add_auth(app, password=args.password, title="Trading Dashboard")

    url = f"http://localhost:{args.port}"
    print(f"\n{'═'*50}")
    print(f"  Dashboard running at  {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'═'*50}\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

"""
screen.py — Full-Universe Stock & Asset Screener
=================================================
Builds a live universe of all major tradable stocks and assets,
runs multi-signal technical screening in parallel, and produces
a ranked BUY / SELL report with scores and reasoning.

Universe (auto-fetched):
  • S&P 500        (~500 US large-cap stocks,   via Wikipedia)
  • NASDAQ 100     (~100 US tech stocks,         via Wikipedia)
  • Dow Jones 30   (30 blue-chip stocks)
  • Popular Mid/Small Caps  (~100 curated)
  • ETFs            (~80 sector/factor/index ETFs)
  • Crypto          (top 15 by volume, via yfinance)
  • Global ADRs     (~50 international stocks)
  Total: ~900 symbols, deduplicated

Screening signals (all computed from OHLCV):
  1. Trend        — price vs SMA50 / SMA200 (Golden Cross)
  2. Momentum     — 12-1 month return (WML factor)
  3. RSI          — oversold / overbought
  4. MACD         — line vs signal crossover
  5. Bollinger    — price vs bands
  6. Volume Surge — volume z-score (breakout confirmation)
  7. Reversal     — short-term contrarian signal
  8. Volatility   — ATR-based trend quality filter

Scoring:
  Each signal votes +1 (bullish) / -1 (bearish) / 0 (neutral).
  Final score = sum of votes. Score ≥ 3 → BUY, ≤ -3 → SELL.
  Confidence = |score| / 8 × 100.

Usage:
  python screen.py                      # full scan, ~5-10 min
  python screen.py --fast               # S&P500 + NASDAQ100 only (~2 min)
  python screen.py --universe sp500     # S&P 500 only
  python screen.py --universe crypto    # crypto only
  python screen.py --universe etf       # ETFs only
  python screen.py --top 30             # show top 30 per side
  python screen.py --min-score 4        # only strong signals
  python screen.py --workers 20         # more parallel workers
  python screen.py --days 180           # shorter lookback
"""

import argparse
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# UNIVERSE DEFINITIONS
# ─────────────────────────────────────────────

DOW_30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]

ETFS = [
    # Broad market
    "SPY","QQQ","IWM","DIA","VTI","VT","ITOT","SCHB",
    # Sectors
    "XLF","XLE","XLK","XLV","XLI","XLU","XLB","XLRE","XLY","XLP","XLC",
    # Factor / Smart Beta
    "MTUM","QUAL","USMV","VTV","VUG","DGRO","NOBL","COWZ",
    # Fixed Income
    "TLT","IEF","SHY","BIL","TIPS","HYG","LQD","AGG","BND","EMB",
    # Commodities
    "GLD","SLV","USO","UNG","PDBC","DBC","CORN","WEAT","SOYB",
    # International
    "EEM","EFA","VEA","IEMG","FXI","EWJ","EWZ","KWEB","EWC","EWA",
    "EWG","EWU","EWH","EWT","EWY","EWS","EWP","EWQ","EWI","EWL",
    # Volatility / Leveraged
    "UVXY","VXX","TQQQ","SQQQ","SPXU","UPRO","SOXL","SOXS",
    # Thematic
    "ARKK","ARKG","ARKW","ARKF","ARKQ","ICLN","TAN","QCLN",
    "BOTZ","ROBO","AIQ","PENX","CIBR","HACK","BUG",
]

CRYPTO = [
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD",
    "AVAX-USD","DOGE-USD","DOT-USD","MATIC-USD","LINK-USD","LTC-USD",
    "UNI7083-USD","ATOM-USD","FIL-USD",
]

GLOBAL_ADRS = [
    # China
    "BABA","JD","PDD","BIDU","NIO","XPEV","LI","TCOM","TME","BILI",
    "FUTU","QFIN","MNSO","YUMC","TAL","EDU","WB","ATHM","GDS",
    # Other Asia
    "TM","HMC","SONY","SE","GRAB","MANU","WIT","INFY","HDB",
    # Europe
    "ASML","SAP","NVO","SHOP","RIO","BP","HSBC","UL","DEO",
    # Americas ex-US
    "MDB","NU","STNE","MELI","SQM","LTM","VALE","PBR",
]

MID_SMALL_CAPS = [
    "PLTR","COIN","HOOD","RIVN","LCID","MSTR","SMCI","SNOW","ARM","APP",
    "SOFI","RBLX","U","AFRM","UPST","OPEN","DKNG","PENN","CRWD","ZS",
    "PANW","NET","MDB","DDOG","GTLB","PATH","CFLT","ESTC","BILL","HUBS",
    "TEAM","ZM","DOCU","OKTA","SPLK","MQ","IONQ","RGTI","QUBT","QBTS",
    "ACHR","JOBY","LILM","EVGO","CHPT","BLNK","WBD","PARA","NFLX","SPOT",
    "ROKU","TTD","APPS","IAS","MGNI","PERI","PUBM","CRTO","TRADE",
    "LYFT","UBER","ABNB","VRBO","BKNG","EXPE","TRIP","DASH","YELP","OPEN",
]

FIXED_UNIVERSE = {
    "dow30":       DOW_30,
    "etf":         ETFS,
    "crypto":      CRYPTO,
    "global_adr":  GLOBAL_ADRS,
    "mid_small":   MID_SMALL_CAPS,
}


# ─────────────────────────────────────────────
# UNIVERSE BUILDER
# ─────────────────────────────────────────────

def fetch_sp500() -> list:
    """Scrape S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].tolist()
        # yfinance uses '-' not '.' for BRK.B etc.
        return [t.replace(".", "-") for t in tickers]
    except Exception as e:
        print(f"  [!] Could not fetch S&P 500 list: {e}")
        return []


def fetch_nasdaq100() -> list:
    """Scrape NASDAQ-100 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            if "Ticker" in t.columns or "Symbol" in t.columns:
                col = "Ticker" if "Ticker" in t.columns else "Symbol"
                return t[col].tolist()
        return []
    except Exception as e:
        print(f"  [!] Could not fetch NASDAQ-100 list: {e}")
        return []


def build_universe(mode: str = "full") -> list:
    """Build deduplicated list of symbols to screen."""
    symbols = []

    if mode in ("full", "sp500"):
        print("  Fetching S&P 500 list...", end=" ", flush=True)
        sp500 = fetch_sp500()
        symbols += sp500
        print(f"{len(sp500)} symbols")

    if mode in ("full", "nasdaq100"):
        print("  Fetching NASDAQ-100 list...", end=" ", flush=True)
        ndx = fetch_nasdaq100()
        symbols += ndx
        print(f"{len(ndx)} symbols")

    if mode in ("full", "fast"):
        for name, lst in FIXED_UNIVERSE.items():
            symbols += lst
            print(f"  Added {name}: {len(lst)} symbols")

    if mode == "etf":
        symbols = ETFS[:]
    elif mode == "crypto":
        symbols = CRYPTO[:]
    elif mode == "dow30":
        symbols = DOW_30[:]

    # Deduplicate preserving order
    seen = set()
    unique = []
    for s in symbols:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


# ─────────────────────────────────────────────
# BATCH DATA DOWNLOADER
# ─────────────────────────────────────────────

def batch_download(symbols: list[str], days: int = 365,
                   chunk_size: int = 100) -> dict:
    """
    Download OHLCV for a list of symbols in chunks using yfinance.
    Returns dict: {symbol: DataFrame}.
    """
    import yfinance as yf

    end   = datetime.today()
    start = end - timedelta(days=days + 10)
    all_data = {}
    total_chunks = (len(symbols) - 1) // chunk_size + 1

    for i, chunk_start in enumerate(range(0, len(symbols), chunk_size)):
        chunk = symbols[chunk_start: chunk_start + chunk_size]
        pct = (i + 1) / total_chunks * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:5.1f}%  downloading chunk {i+1}/{total_chunks}  ",
              end="", flush=True)
        try:
            raw = yf.download(
                chunk, start=start, end=end,
                progress=False, auto_adjust=True, group_by="ticker",
                threads=True
            )
        except Exception as e:
            print(f"\n  [!] Download error chunk {i+1}: {e}")
            continue

        if raw.empty:
            continue

        # Parse per-ticker DataFrames out of multi-level columns
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance v1.x: level 0 = ticker, level 1 = field (Open/Close/…)
            lvl0 = raw.columns.get_level_values(0).unique().tolist()
            lvl1 = raw.columns.get_level_values(1).unique().tolist()
            # detect which level holds tickers vs fields
            fields = {"open","high","low","close","volume","adj close"}
            ticker_level = 1 if set(l.lower() for l in lvl0) <= fields else 0
            for sym in chunk:
                try:
                    df = raw.xs(sym, axis=1, level=ticker_level).copy()
                    df.columns = [c.lower() for c in df.columns]
                    df = df.dropna(subset=["close"])
                    if len(df) >= 50:
                        all_data[sym] = df
                except Exception:
                    pass
        else:
            # Single ticker returned as flat frame
            if len(chunk) == 1:
                df = raw.copy()
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                if len(df) >= 50:
                    all_data[chunk[0]] = df

    print(f"\r  [{'█'*20}] 100.0%  Downloaded {len(all_data)}/{len(symbols)} symbols  ")
    return all_data


# ─────────────────────────────────────────────
# SIGNAL FUNCTIONS (vectorized, fast)
# ─────────────────────────────────────────────

def _s(c, w):    return c.rolling(w).mean()
def _e(c, sp):   return c.ewm(span=sp, adjust=False).mean()
def _rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def compute_signals(df: pd.DataFrame) -> dict:
    """
    Compute 8 screening signals. Returns dict with each signal value
    (+1 bullish / -1 bearish / 0 neutral) and its label.
    """
    c   = df["close"].astype(float)
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)
    n   = len(c)

    signals = {}

    # 1 ─ Trend: price vs SMA50 / SMA200 (Golden/Death Cross)
    if n >= 200:
        s50, s200 = _s(c, 50).iloc[-1], _s(c, 200).iloc[-1]
        s50p, s200p = _s(c, 50).iloc[-2], _s(c, 200).iloc[-2]
        price = c.iloc[-1]
        if price > s50 > s200:
            signals["trend"] = (+1, "Price > SMA50 > SMA200 (uptrend)")
        elif price < s50 < s200:
            signals["trend"] = (-1, "Price < SMA50 < SMA200 (downtrend)")
        elif s50p < s200p and s50 > s200:
            signals["trend"] = (+1, "Golden Cross just triggered")
        elif s50p > s200p and s50 < s200:
            signals["trend"] = (-1, "Death Cross just triggered")
        else:
            signals["trend"] = (0, "Mixed trend")
    elif n >= 50:
        s50 = _s(c, 50).iloc[-1]
        signals["trend"] = (+1 if c.iloc[-1] > s50 else -1, "vs SMA50")
    else:
        signals["trend"] = (0, "Insufficient data")

    # 2 ─ Momentum: 12-1 month return
    months_12 = min(252, n - 1)
    months_skip = min(21, n - 2)
    if n > months_skip + 2:
        mom = (c.iloc[-(months_skip + 1)] / c.iloc[-months_12]) - 1 if n >= months_12 else 0
        if mom > 0.10:
            signals["momentum"] = (+1, f"12-1mo return: +{mom*100:.1f}%")
        elif mom < -0.10:
            signals["momentum"] = (-1, f"12-1mo return: {mom*100:.1f}%")
        else:
            signals["momentum"] = (0, f"12-1mo return: {mom*100:.1f}% (weak)")
    else:
        signals["momentum"] = (0, "Insufficient data")

    # 3 ─ RSI
    if n >= 16:
        rsi = float(_rsi(c).iloc[-1])
        if rsi < 30:
            signals["rsi"] = (+1, f"RSI={rsi:.1f} (oversold)")
        elif rsi > 70:
            signals["rsi"] = (-1, f"RSI={rsi:.1f} (overbought)")
        elif rsi < 45:
            signals["rsi"] = (+1, f"RSI={rsi:.1f} (bullish zone)")
        elif rsi > 55:
            signals["rsi"] = (-1, f"RSI={rsi:.1f} (bearish zone)")
        else:
            signals["rsi"] = (0, f"RSI={rsi:.1f} (neutral)")
    else:
        signals["rsi"] = (0, "Insufficient data")

    # 4 ─ MACD crossover
    if n >= 35:
        macd_l = _e(c, 12) - _e(c, 26)
        macd_s = macd_l.ewm(span=9, adjust=False).mean()
        cross = (macd_l.iloc[-2] <= macd_s.iloc[-2] and macd_l.iloc[-1] > macd_s.iloc[-1])
        dcross = (macd_l.iloc[-2] >= macd_s.iloc[-2] and macd_l.iloc[-1] < macd_s.iloc[-1])
        above = macd_l.iloc[-1] > macd_s.iloc[-1]
        if cross:
            signals["macd"] = (+1, "MACD bullish crossover")
        elif dcross:
            signals["macd"] = (-1, "MACD bearish crossover")
        elif above:
            signals["macd"] = (+1, "MACD line above signal")
        else:
            signals["macd"] = (-1, "MACD line below signal")
    else:
        signals["macd"] = (0, "Insufficient data")

    # 5 ─ Bollinger Bands
    if n >= 22:
        mid  = c.rolling(20).mean()
        std_ = c.rolling(20).std()
        upper, lower = mid + 2*std_, mid - 2*std_
        p = c.iloc[-1]
        bb_pos = (p - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-10)
        if p < lower.iloc[-1]:
            signals["bollinger"] = (+1, f"Below lower band (BB%={bb_pos:.2f})")
        elif p > upper.iloc[-1]:
            signals["bollinger"] = (-1, f"Above upper band (BB%={bb_pos:.2f})")
        elif bb_pos < 0.35:
            signals["bollinger"] = (+1, f"Near lower band (BB%={bb_pos:.2f})")
        elif bb_pos > 0.65:
            signals["bollinger"] = (-1, f"Near upper band (BB%={bb_pos:.2f})")
        else:
            signals["bollinger"] = (0, f"Mid band (BB%={bb_pos:.2f})")
    else:
        signals["bollinger"] = (0, "Insufficient data")

    # 6 ─ Volume Surge (breakout confirmation)
    if not vol.empty and n >= 21:
        vol_ma = float(vol.rolling(20).mean().iloc[-1])
        vol_now = float(vol.iloc[-1])
        vol_z = (vol_now - vol_ma) / (vol.rolling(20).std().iloc[-1] + 1e-10)
        price_ret = float(c.pct_change().iloc[-1])
        if vol_z > 1.5 and price_ret > 0:
            signals["volume"] = (+1, f"Volume surge +{vol_z:.1f}σ with up move")
        elif vol_z > 1.5 and price_ret < 0:
            signals["volume"] = (-1, f"Volume surge +{vol_z:.1f}σ with down move")
        elif vol_z < -1.0:
            signals["volume"] = (0, f"Volume drying up ({vol_z:.1f}σ)")
        else:
            signals["volume"] = (0, f"Volume normal ({vol_z:.1f}σ)")
    else:
        signals["volume"] = (0, "No volume data")

    # 7 ─ Short-Term Reversal
    if n >= 7:
        r5  = float((c.iloc[-1] / c.iloc[-min(5, n-1)]) - 1)
        rsi7 = float(_rsi(c, 7).iloc[-1]) if n >= 9 else 50
        if r5 < -0.05 and rsi7 < 35:
            signals["reversal"] = (+1, f"5d drop {r5*100:.1f}% + RSI={rsi7:.0f} (bounce setup)")
        elif r5 > 0.05 and rsi7 > 65:
            signals["reversal"] = (-1, f"5d rally {r5*100:.1f}% + RSI={rsi7:.0f} (fade setup)")
        else:
            signals["reversal"] = (0, f"5d ret={r5*100:.1f}%")
    else:
        signals["reversal"] = (0, "Insufficient data")

    # 8 ─ Volatility / ATR trend quality
    if n >= 15:
        hi, lo, cl = df["high"].astype(float), df["low"].astype(float), c
        tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr / float(c.iloc[-1]) * 100
        rv = float(c.pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) * 100 if n >= 22 else 0
        # Low ATR in uptrend = healthy; very high ATR = risk
        trend_dir = signals.get("trend", (0,))[0]
        if atr_pct < 2.0 and trend_dir == 1:
            signals["volatility"] = (+1, f"Low ATR {atr_pct:.2f}% — steady uptrend")
        elif atr_pct > 5.0 and trend_dir == -1:
            signals["volatility"] = (-1, f"High ATR {atr_pct:.2f}% — chaotic downtrend")
        else:
            signals["volatility"] = (0, f"ATR={atr_pct:.2f}% RV={rv:.0f}%")
    else:
        signals["volatility"] = (0, "Insufficient data")

    return signals


# ─────────────────────────────────────────────
# SCREEN SINGLE ASSET
# ─────────────────────────────────────────────

def screen_asset(symbol: str, df: pd.DataFrame) -> dict:
    """Run all signals on one asset, return scored result."""
    try:
        sigs = compute_signals(df)
    except Exception as e:
        return None

    votes     = [v for v, _ in sigs.values()]
    buy_cnt   = sum(1 for v in votes if v == 1)
    sell_cnt  = sum(1 for v in votes if v == -1)
    net_score = sum(votes)
    conf      = round(abs(net_score) / len(sigs) * 100, 1)

    close     = df["close"].astype(float)
    price     = round(float(close.iloc[-1]), 4)
    ret_1d    = round(float(close.pct_change().iloc[-1]) * 100, 2)
    ret_5d    = round(float((close.iloc[-1] / close.iloc[min(-5, -(len(close)-1))]) - 1) * 100, 2)
    ret_1mo   = round(float((close.iloc[-1] / close.iloc[min(-21, -(len(close)-1))]) - 1) * 100, 2)
    ret_3mo   = round(float((close.iloc[-1] / close.iloc[min(-63, -(len(close)-1))]) - 1) * 100, 2)
    vol_20    = round(float(close.pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) * 100, 1)

    if net_score >= 3:
        signal = "BUY"
    elif net_score <= -3:
        signal = "SELL"
    elif net_score >= 2:
        signal = "WEAK BUY"
    elif net_score <= -2:
        signal = "WEAK SELL"
    else:
        signal = "NEUTRAL"

    # Build reason string from top contributing signals
    reasons = [label for v, label in sigs.values() if v != 0][:3]
    reason  = " | ".join(reasons)

    return {
        "symbol":    symbol,
        "price":     price,
        "signal":    signal,
        "score":     net_score,
        "confidence":conf,
        "buy_votes": buy_cnt,
        "sell_votes":sell_cnt,
        "ret_1d":    ret_1d,
        "ret_5d":    ret_5d,
        "ret_1mo":   ret_1mo,
        "ret_3mo":   ret_3mo,
        "vol_ann":   vol_20,
        "reason":    reason,
        # Individual signal votes
        **{f"sig_{k}": v for k, (v, _) in sigs.items()},
    }


# ─────────────────────────────────────────────
# PARALLEL SCREENER
# ─────────────────────────────────────────────

def run_screener(data: dict, workers: int = 12) -> pd.DataFrame:
    """Screen all downloaded assets in parallel threads."""
    results = []
    total   = len(data)
    done    = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(screen_asset, sym, df): sym
                   for sym, df in data.items()}
        for fut in as_completed(futures):
            done += 1
            pct  = done / total * 100
            bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:5.1f}%  screening {futures[fut]:<14}",
                  end="", flush=True)
            row = fut.result()
            if row:
                results.append(row)

    print(f"\r  [{'█'*20}] 100.0%  Screened {len(results)}/{total} assets  ")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────

COLORS = {
    "BUY":       "\033[92m",   # green
    "SELL":      "\033[91m",   # red
    "WEAK BUY":  "\033[96m",   # cyan
    "WEAK SELL": "\033[93m",   # yellow
    "NEUTRAL":   "\033[90m",   # gray
    "RESET":     "\033[0m",
}

def _color(text, key):
    c = COLORS.get(key, "")
    r = COLORS["RESET"]
    return f"{c}{text}{r}"


def print_report(df: pd.DataFrame, top_n: int = 30, min_score: int = 3,
                 signal_filter: str = None, no_color: bool = False):
    if df.empty:
        print("  No results.")
        return

    def clr(text, key):
        return text if no_color else _color(text, key)

    # Separate by signal
    strong_buy  = df[df["signal"] == "BUY"].sort_values("score", ascending=False)
    weak_buy    = df[df["signal"] == "WEAK BUY"].sort_values("score", ascending=False)
    strong_sell = df[df["signal"] == "SELL"].sort_values("score")
    weak_sell   = df[df["signal"] == "WEAK SELL"].sort_values("score")
    neutral     = df[df["signal"] == "NEUTRAL"]

    total_signals = len(strong_buy) + len(weak_buy) + len(strong_sell) + len(weak_sell)

    ts = datetime.today().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'═'*90}")
    print(f"  SCREENER REPORT — {ts}")
    print(f"  {len(df)} assets screened  |  "
          f"STRONG BUY: {len(strong_buy)}  WEAK BUY: {len(weak_buy)}  "
          f"NEUTRAL: {len(neutral)}  WEAK SELL: {len(weak_sell)}  STRONG SELL: {len(strong_sell)}")
    print(f"{'═'*90}")

    header = (f"  {'#':>3}  {'Symbol':<12} {'Price':>10} {'Score':>6} {'Conf%':>6} "
              f"{'1d%':>6} {'5d%':>6} {'1mo%':>7} {'3mo%':>7} {'Vol%':>6}  Reason")
    sep    = "  " + "─" * 87

    def _block(subset, label, key, limit):
        if subset.empty or (signal_filter and signal_filter.upper() not in label.upper()):
            return
        shown = subset.head(limit)
        print(f"\n  {clr('──', key)} {clr(label, key)} "
              f"({len(subset)} total, showing top {min(limit, len(subset))}) {clr('──', key)}\n")
        print(header)
        print(sep)
        for rank, (_, row) in enumerate(shown.iterrows(), 1):
            sym_str  = clr(f"{row['symbol']:<12}", key)
            sig_str  = clr(f"{row['signal']:<10}", key)
            print(
                f"  {rank:>3}  {sym_str} {row['price']:>10.4f} "
                f"{row['score']:>+6.0f} {row['confidence']:>6.1f} "
                f"{row['ret_1d']:>+6.2f} {row['ret_5d']:>+6.2f} "
                f"{row['ret_1mo']:>+7.2f} {row['ret_3mo']:>+7.2f} "
                f"{row['vol_ann']:>6.1f}  {row['reason'][:55]}"
            )

    _block(strong_buy,  "★ STRONG BUY  (score ≥ +3)",  "BUY",       top_n)
    _block(weak_buy,    "◑ WEAK BUY   (score = +2)",    "WEAK BUY",  top_n // 2)
    _block(strong_sell, "★ STRONG SELL (score ≤ -3)",   "SELL",      top_n)
    _block(weak_sell,   "◑ WEAK SELL  (score = -2)",    "WEAK SELL", top_n // 2)

    if not signal_filter:
        print(f"\n  {clr('── NEUTRAL', 'NEUTRAL')} — {len(neutral)} assets (score -1 to +1, omitted from report)")

    print(f"\n{'═'*90}")
    print(f"  Score key: +1 per bullish signal, -1 per bearish signal (8 signals total)")
    print(f"  Signals: Trend | Momentum | RSI | MACD | Bollinger | Volume | Reversal | ATR")
    print(f"{'═'*90}\n")


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────

def plot_report(df: pd.DataFrame, top_n: int = 20, save_path: str = "screen_report.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  pip install matplotlib for charts")
        return

    actionable = df[df["signal"].isin(["BUY","WEAK BUY","SELL","WEAK SELL"])]
    actionable = actionable.sort_values("score", ascending=False)
    top_buy    = actionable[actionable["score"] > 0].head(top_n // 2)
    top_sell   = actionable[actionable["score"] < 0].tail(top_n // 2)
    plot_df    = pd.concat([top_buy, top_sell]).reset_index(drop=True)

    if plot_df.empty:
        return

    colors = ["#27ae60" if s > 0 else "#e74c3c" for s in plot_df["score"]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # ── Bar: Score ────────────────────────────
    axes[0].barh(plot_df["symbol"][::-1], plot_df["score"][::-1],
                 color=colors[::-1], alpha=0.85, edgecolor="white", height=0.7)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].axvline(3,  color="#27ae60", lw=1, ls="--", alpha=0.5)
    axes[0].axvline(-3, color="#e74c3c", lw=1, ls="--", alpha=0.5)
    axes[0].set_xlabel("Signal Score")
    axes[0].set_title("Signal Score (max ±8)", fontweight="bold")
    buy_patch  = mpatches.Patch(color="#27ae60", label="BUY")
    sell_patch = mpatches.Patch(color="#e74c3c", label="SELL")
    axes[0].legend(handles=[buy_patch, sell_patch])

    # ── Scatter: Return vs Score ───────────────
    buy_df  = df[df["score"] > 0]
    sell_df = df[df["score"] < 0]
    neut_df = df[df["score"] == 0]
    axes[1].scatter(neut_df["ret_1mo"], neut_df["score"],  c="gray",    s=25, alpha=0.3, label="Neutral")
    axes[1].scatter(buy_df["ret_1mo"],  buy_df["score"],   c="#27ae60", s=40, alpha=0.7, label="Buy")
    axes[1].scatter(sell_df["ret_1mo"], sell_df["score"],  c="#e74c3c", s=40, alpha=0.7, label="Sell")
    for _, row in plot_df.head(8).iterrows():
        axes[1].annotate(row["symbol"], (row["ret_1mo"], row["score"]),
                         fontsize=7, xytext=(2, 2), textcoords="offset points")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].axvline(0, color="black", lw=0.5, ls="--")
    axes[1].set_xlabel("1-Month Return (%)")
    axes[1].set_ylabel("Signal Score")
    axes[1].set_title("Signal Score vs 1-Month Return", fontweight="bold")
    axes[1].legend()

    # ── Heatmap: individual signals ────────────
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    heat_df  = plot_df[["symbol"] + sig_cols].set_index("symbol")
    heat_num = heat_df.astype(float)
    im = axes[2].imshow(heat_num.values.T, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_xticks(range(len(heat_num.index)))
    axes[2].set_yticks(range(len(sig_cols)))
    axes[2].set_xticklabels(heat_num.index, rotation=45, ha="right", fontsize=7)
    axes[2].set_yticklabels([c.replace("sig_", "") for c in sig_cols])
    plt.colorbar(im, ax=axes[2], label="-1=Bearish  0=Neutral  +1=Bullish")
    axes[2].set_title("Signal Breakdown Heatmap", fontweight="bold")
    for i in range(len(heat_num.index)):
        for j in range(len(sig_cols)):
            v = int(heat_num.iloc[i, j])
            axes[2].text(i, j, {1: "▲", -1: "▼", 0: "─"}.get(v, ""),
                         ha="center", va="center", fontsize=8, color="black")

    plt.suptitle(f"Stock Screener Report — {datetime.today().strftime('%Y-%m-%d')}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Screen all tradable stocks & assets")
    parser.add_argument("--universe", default="full",
                        choices=["full","fast","sp500","nasdaq100","etf","crypto","dow30","global"],
                        help="Which universe to screen (default: full ~900 symbols)")
    parser.add_argument("--days",      type=int, default=365,  help="Lookback window in days")
    parser.add_argument("--top",       type=int, default=30,   help="Show top N per signal side")
    parser.add_argument("--min-score", type=int, default=2,    help="Minimum |score| to include")
    parser.add_argument("--signal",    choices=["BUY","SELL"], help="Filter report to one side")
    parser.add_argument("--workers",   type=int, default=12,   help="Parallel screening threads")
    parser.add_argument("--no-save",   action="store_true",    help="Skip CSV output")
    parser.add_argument("--no-chart",  action="store_true",    help="Skip chart output")
    parser.add_argument("--no-color",  action="store_true",    help="Plain text (no ANSI colors)")
    args = parser.parse_args()

    t0 = time.time()

    # ── 1. Build universe ───────────────────────
    print(f"\n{'═'*65}")
    print(f"  STOCK SCREENER  —  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Universe: {args.universe.upper()}   Lookback: {args.days}d   Workers: {args.workers}")
    print(f"{'═'*65}")
    print("\n  Building universe...")
    symbols = build_universe(args.universe)
    print(f"  → {len(symbols)} unique symbols to screen\n")

    # ── 2. Batch download ───────────────────────
    print("  Downloading price data...")
    data = batch_download(symbols, days=args.days)
    print(f"  → {len(data)} assets with sufficient data\n")

    # ── 3. Screen ───────────────────────────────
    print("  Running screening signals...")
    results = run_screener(data, workers=args.workers)
    print(f"  → {len(results)} assets screened\n")

    # ── 4. Report ──────────────────────────────
    print_report(results, top_n=args.top, min_score=args.min_score,
                 signal_filter=args.signal, no_color=args.no_color)

    elapsed = time.time() - t0
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")

    # ── 5. Save CSV ─────────────────────────────
    if not args.no_save:
        fname = f"screen_{args.universe}_{datetime.today().strftime('%Y%m%d_%H%M')}.csv"
        results.to_csv(fname, index=False)
        print(f"  Full results saved → {fname}")

    # ── 6. Chart ────────────────────────────────
    if not args.no_chart:
        chart_path = f"screen_{args.universe}_{datetime.today().strftime('%Y%m%d')}.png"
        plot_report(results, top_n=args.top, save_path=chart_path)


if __name__ == "__main__":
    main()

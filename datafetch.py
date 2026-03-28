"""
Market Data Hub — Multi-provider data fetcher + daily trading signal scanner.

Supported providers:
  - yfinance    : US stocks, ETFs, FX, crypto (daily/intraday)
  - akshare     : China A-shares, futures, indices (free, no account)
  - binance     : Crypto OHLCV via public REST (no account needed)
  - tushare     : China A-shares / futures (requires free token)
  - fredapi     : Macro/economic indicators (requires free API key)

Strategies (signal generators):
  - MA Crossover  : SMA20 crosses SMA50
  - RSI           : Oversold (<30) / Overbought (>70)
  - MACD          : MACD line crosses signal line
  - Bollinger     : Price breaks above/below bands

Usage:
  python datafetch.py                  # run daily scan on default watchlist
  python datafetch.py --query AAPL     # fetch + show OHLCV for a symbol
  python datafetch.py --provider all   # scan all providers
  python datafetch.py --list           # show all configured symbols
"""

import argparse
import warnings
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit your watchlists and API keys here
# ─────────────────────────────────────────────

TUSHARE_TOKEN = ""          # get free token at tushare.pro
FRED_API_KEY  = ""          # get free key at fred.stlouisfed.org

WATCHLIST = {
    "us_stocks": ["AAPL", "TSLA", "NVDA", "SPY", "QQQ"],
    "crypto":    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
    "cn_stocks": ["000001.SZ", "600519.SH", "300750.SZ"],   # tushare format
    "cn_futures":["cu888", "rb888", "i888"],                 # akshare format
    "macro":     ["DGS10", "CPIAUCSL", "FEDFUNDS"],          # FRED series
}

DEFAULT_DAYS = 120   # lookback window for signal calculation


# ─────────────────────────────────────────────
# PROVIDER LAYER
# ─────────────────────────────────────────────

def fetch_yfinance(symbol: str, days: int = DEFAULT_DAYS) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Yahoo Finance. Covers US/HK stocks, FX, ETFs, crypto."""
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        # yfinance v1.x returns MultiIndex columns (Price, Ticker) — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        return df[["open", "close", "high", "low", "volume"]]
    except ImportError:
        print("  [!] yfinance not installed: pip install yfinance")
        return None
    except Exception as e:
        print(f"  [!] yfinance error for {symbol}: {e}")
        return None


def fetch_binance(symbol: str, days: int = DEFAULT_DAYS,
                  interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Binance public REST API. No account needed."""
    try:
        import requests
        limit = min(days, 1000)
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
        ])
        df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        print(f"  [!] Binance error for {symbol}: {e}")
        return None


def fetch_akshare_stock(symbol: str, days: int = DEFAULT_DAYS) -> Optional[pd.DataFrame]:
    """Fetch A-share daily OHLCV from AKShare. symbol e.g. '000001' (no suffix)."""
    try:
        import akshare as ak
        end   = datetime.today().strftime("%Y%m%d")
        start = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
        # strip exchange suffix if present
        code = symbol.split(".")[0]
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                 start_date=start, end_date=end, adjust="qfq")
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        return df
    except ImportError:
        print("  [!] akshare not installed: pip install akshare")
        return None
    except Exception as e:
        print(f"  [!] AKShare stock error for {symbol}: {e}")
        return None


def fetch_akshare_futures(symbol: str, days: int = DEFAULT_DAYS) -> Optional[pd.DataFrame]:
    """Fetch Chinese futures daily OHLCV from AKShare. symbol e.g. 'cu888'."""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=symbol, adjust="")
        df = df.rename(columns={
            "日期": "date", "开盘价": "open", "收盘价": "close",
            "最高价": "high", "最低价": "low", "成交量": "volume"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].tail(days)
        return df
    except ImportError:
        print("  [!] akshare not installed: pip install akshare")
        return None
    except Exception as e:
        print(f"  [!] AKShare futures error for {symbol}: {e}")
        return None


def fetch_tushare(symbol: str, days: int = DEFAULT_DAYS) -> Optional[pd.DataFrame]:
    """Fetch A-share / futures daily OHLCV from Tushare Pro. Requires token."""
    if not TUSHARE_TOKEN:
        print("  [!] Tushare token not set. Edit TUSHARE_TOKEN in datafetch.py")
        return None
    try:
        import tushare as ts
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        end   = datetime.today().strftime("%Y%m%d")
        start = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
        df = pro.daily(ts_code=symbol, start_date=start, end_date=end)
        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]]
    except ImportError:
        print("  [!] tushare not installed: pip install tushare")
        return None
    except Exception as e:
        print(f"  [!] Tushare error for {symbol}: {e}")
        return None


def fetch_fred(series_id: str, days: int = DEFAULT_DAYS) -> Optional[pd.DataFrame]:
    """Fetch macro time series from FRED. Requires free API key."""
    if not FRED_API_KEY:
        print("  [!] FRED API key not set. Edit FRED_API_KEY in datafetch.py")
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        start = datetime.today() - timedelta(days=days)
        s = fred.get_series(series_id, observation_start=start)
        df = s.to_frame(name="close").dropna()
        df.index.name = "date"
        return df
    except ImportError:
        print("  [!] fredapi not installed: pip install fredapi")
        return None
    except Exception as e:
        print(f"  [!] FRED error for {series_id}: {e}")
        return None


# ─────────────────────────────────────────────
# AUTO-ROUTING: pick provider by symbol format
# ─────────────────────────────────────────────

def fetch(symbol: str, provider: str = "auto", days: int = DEFAULT_DAYS) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV for any symbol, auto-detecting the right provider.
    Override with provider='yfinance'|'binance'|'akshare'|'tushare'|'fred'
    """
    if provider == "binance" or (provider == "auto" and symbol.endswith("USDT")):
        return fetch_binance(symbol, days)

    if provider == "fred" or (provider == "auto" and symbol.isupper() and len(symbol) > 5 and "." not in symbol):
        return fetch_fred(symbol, days)

    if provider == "tushare" or (provider == "auto" and symbol.endswith((".SH", ".SZ", ".BJ"))):
        return fetch_tushare(symbol, days)

    if provider == "akshare_futures" or (provider == "auto" and symbol[-3:].isdigit() and symbol[:2].isalpha()):
        return fetch_akshare_futures(symbol, days)

    if provider == "akshare" or (provider == "auto" and symbol.isdigit()):
        return fetch_akshare_stock(symbol, days)

    # default: yfinance (US stocks, ETFs, FX, indices)
    return fetch_yfinance(symbol, days)


# ─────────────────────────────────────────────
# STRATEGY / SIGNAL ENGINE
# ─────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def _bollinger(series: pd.Series, window=20, num_std=2):
    mid  = series.rolling(window).mean()
    std  = series.rolling(window).std()
    return mid + num_std * std, mid - num_std * std  # upper, lower


def generate_signals(df: pd.DataFrame) -> dict:
    """
    Run all strategies on a DataFrame and return a dict of signals.
    Returns: { 'ma_cross': 'BUY'|'SELL'|None, 'rsi': ..., 'macd': ..., 'bollinger': ... }
    """
    signals = {}
    close = df["close"].astype(float)

    if len(close) < 55:
        return {"error": "not enough data (need 55+ bars)"}

    # ── MA Crossover (SMA20 / SMA50) ──────────────────────
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    if sma20.iloc[-2] <= sma50.iloc[-2] and sma20.iloc[-1] > sma50.iloc[-1]:
        signals["ma_cross"] = "BUY"
    elif sma20.iloc[-2] >= sma50.iloc[-2] and sma20.iloc[-1] < sma50.iloc[-1]:
        signals["ma_cross"] = "SELL"
    else:
        signals["ma_cross"] = None

    # ── RSI ───────────────────────────────────────────────
    rsi = _rsi(close).iloc[-1]
    signals["rsi_value"] = round(rsi, 1)
    if rsi < 30:
        signals["rsi"] = "BUY (oversold)"
    elif rsi > 70:
        signals["rsi"] = "SELL (overbought)"
    else:
        signals["rsi"] = None

    # ── MACD ─────────────────────────────────────────────
    macd_line, signal_line = _macd(close)
    if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals["macd"] = "BUY"
    elif macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        signals["macd"] = "SELL"
    else:
        signals["macd"] = None

    # ── Bollinger Band Breakout ───────────────────────────
    upper, lower = _bollinger(close)
    price = close.iloc[-1]
    if price < lower.iloc[-1]:
        signals["bollinger"] = "BUY (below lower band)"
    elif price > upper.iloc[-1]:
        signals["bollinger"] = "SELL (above upper band)"
    else:
        signals["bollinger"] = None

    # ── Composite Score ───────────────────────────────────
    buy_count  = sum(1 for k, v in signals.items() if isinstance(v, str) and "BUY" in v)
    sell_count = sum(1 for k, v in signals.items() if isinstance(v, str) and "SELL" in v)
    if buy_count >= 2:
        signals["composite"] = f"STRONG BUY  ({buy_count}/4 signals)"
    elif sell_count >= 2:
        signals["composite"] = f"STRONG SELL ({sell_count}/4 signals)"
    elif buy_count == 1:
        signals["composite"] = f"WEAK BUY    ({buy_count}/4 signals)"
    elif sell_count == 1:
        signals["composite"] = f"WEAK SELL   ({sell_count}/4 signals)"
    else:
        signals["composite"] = "NEUTRAL"

    return signals


# ─────────────────────────────────────────────
# SCANNER — daily opportunity scan
# ─────────────────────────────────────────────

def scan_opportunities(watchlist: dict = None, days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """
    Scan all symbols in the watchlist, generate signals, return a summary DataFrame.
    """
    if watchlist is None:
        watchlist = WATCHLIST

    results = []
    all_symbols = []
    for category, symbols in watchlist.items():
        if category == "macro":
            continue   # FRED series don't have OHLCV for strategy signals
        for sym in symbols:
            all_symbols.append((category, sym))

    print(f"\n{'='*60}")
    print(f"  DAILY OPPORTUNITY SCAN  —  {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  Scanning {len(all_symbols)} symbols across {len(watchlist)-1} categories")
    print(f"{'='*60}")

    for category, symbol in all_symbols:
        print(f"  Fetching {symbol:<18} [{category}]", end=" ... ", flush=True)
        df = fetch(symbol, days=days)
        if df is None or df.empty:
            print("NO DATA")
            continue
        signals = generate_signals(df)
        if "error" in signals:
            print(signals["error"])
            continue

        price = df["close"].iloc[-1]
        print(signals["composite"])

        results.append({
            "symbol":    symbol,
            "category":  category,
            "price":     round(float(price), 4),
            "rsi":       signals.get("rsi_value"),
            "ma_cross":  signals.get("ma_cross") or "-",
            "macd":      signals.get("macd") or "-",
            "bollinger": signals.get("bollinger") or "-",
            "signal":    signals.get("composite"),
        })

    df_results = pd.DataFrame(results)
    return df_results


def print_report(df: pd.DataFrame):
    """Print a formatted trading opportunity report."""
    if df.empty:
        print("\n  No results to display.")
        return

    print(f"\n{'='*60}")
    print("  TRADING SIGNALS SUMMARY")
    print(f"{'='*60}")

    for label, keyword in [("BUY OPPORTUNITIES", "BUY"), ("SELL OPPORTUNITIES", "SELL")]:
        subset = df[df["signal"].str.contains(keyword, na=False)]
        if subset.empty:
            continue
        print(f"\n  ── {label} ──")
        for _, row in subset.iterrows():
            print(f"  {row['symbol']:<18} {row['signal']:<28} "
                  f"price={row['price']}  RSI={row['rsi']}")

    neutral = df[df["signal"] == "NEUTRAL"]
    print(f"\n  ── NEUTRAL ({len(neutral)} symbols) ──")
    for _, row in neutral.iterrows():
        print(f"  {row['symbol']:<18} RSI={row['rsi']}")

    print(f"\n{'='*60}")
    print(f"  Scan complete. {len(df)} symbols processed.")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Market Data Hub — fetch data and scan for trading opportunities"
    )
    parser.add_argument("--query",    metavar="SYMBOL", help="Fetch and display OHLCV for a single symbol")
    parser.add_argument("--signal",   metavar="SYMBOL", help="Show strategy signals for a single symbol")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "yfinance", "binance", "akshare",
                                 "akshare_futures", "tushare", "fred"],
                        help="Force a specific data provider")
    parser.add_argument("--days",     type=int, default=DEFAULT_DAYS, help="Lookback window in days")
    parser.add_argument("--list",     action="store_true", help="List all configured watchlist symbols")
    parser.add_argument("--scan",     action="store_true", help="Run full opportunity scan (default action)")
    parser.add_argument("--save",     metavar="FILE", help="Save scan results to CSV file")
    args = parser.parse_args()

    # ── List watchlist ──────────────────────────────────
    if args.list:
        print("\nConfigured watchlist:")
        for category, symbols in WATCHLIST.items():
            print(f"  [{category}] {', '.join(symbols)}")
        return

    # ── Single symbol query ─────────────────────────────
    if args.query:
        print(f"\nFetching {args.query} via {args.provider} ({args.days} days)...")
        df = fetch(args.query, provider=args.provider, days=args.days)
        if df is not None and not df.empty:
            print(df.tail(20).to_string())
        else:
            print("No data returned.")
        return

    # ── Single symbol signals ───────────────────────────
    if args.signal:
        print(f"\nGenerating signals for {args.signal}...")
        df = fetch(args.signal, provider=args.provider, days=args.days)
        if df is None or df.empty:
            print("No data returned.")
            return
        signals = generate_signals(df)
        print(f"\n  Symbol  : {args.signal}")
        print(f"  Price   : {df['close'].iloc[-1]:.4f}")
        for k, v in signals.items():
            if v:
                print(f"  {k:<12}: {v}")
        return

    # ── Full scan (default) ─────────────────────────────
    results = scan_opportunities(days=args.days)
    print_report(results)

    if args.save and not results.empty:
        results.to_csv(args.save, index=False)
        print(f"  Results saved to {args.save}")


if __name__ == "__main__":
    main()

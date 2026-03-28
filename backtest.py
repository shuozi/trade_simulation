"""
backtest.py — Walk-forward backtest on 100 most-traded assets
=============================================================
For each asset: runs all strategy signals on a rolling window,
tracks actual next-period returns, computes performance metrics,
then generates a CURRENT live signal ranked by backtest quality.

Output: ranked BUY / SELL opportunities with confidence score.

Usage:
    python backtest.py                     # full scan, save to results.csv
    python backtest.py --top 20            # show top 20 opportunities only
    python backtest.py --signal BUY        # show only buy recommendations
    python backtest.py --signal SELL       # show only sell recommendations
    python backtest.py --asset AAPL        # backtest a single asset
    python backtest.py --days 180          # use 180-day history
    python backtest.py --no-save           # skip CSV output
"""

import argparse
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 100 MOST TRADED ASSETS (by avg daily volume)
# ─────────────────────────────────────────────

ASSETS = {
    "us_mega_cap": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
        "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST",
        "ABBV", "MRK", "BAC", "CVX", "KO", "PEP", "LLY", "WMT",
    ],
    "us_growth": [
        "AMD", "PLTR", "SNOW", "MSTR", "SMCI", "ARM", "APP", "HOOD",
        "COIN", "SOFI", "RIVN", "LCID", "NIO", "BABA", "JD",
    ],
    "us_sector": [
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLB", "XLRE",
        "GDX", "SLV", "GLD", "USO", "TLT", "HYG", "LQD",
    ],
    "etf_indices": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "EEM", "EFA",
        "SQQQ", "TQQQ", "SPXU", "UVXY",
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
        "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD",
    ],
    "global_etf": [
        "FXI", "EWJ", "EWZ", "KWEB", "IEMG", "VEA", "EWC", "EWA",
    ],
    "rates_macro": [
        "TLT", "IEF", "SHY", "BIL", "TIPS",
    ],
}

# Flatten to list of 100, deduplicate
ALL_ASSETS = []
seen = set()
for group in ASSETS.values():
    for sym in group:
        if sym not in seen:
            ALL_ASSETS.append(sym)
            seen.add(sym)

# ─────────────────────────────────────────────
# STRATEGIES (self-contained, no import needed)
# ─────────────────────────────────────────────

def _sma(s, w): return s.rolling(w).mean()
def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
def _macd(s, f=12, sl=26, sig=9):
    line = _ema(s, f) - _ema(s, sl)
    return line, line.ewm(span=sig, adjust=False).mean()
def _bb(s, w=20, std=2):
    mid = s.rolling(w).mean()
    sd  = s.rolling(w).std()
    return mid + std*sd, mid, mid - std*sd


STRATEGIES = {
    # ── Momentum ──────────────────────────────
    "TSMOM": lambda df: _tsmom(df),
    "MA_Cross": lambda df: _ma_cross(df),
    "Reversal": lambda df: _reversal(df),
    # ── Mean Reversion ────────────────────────
    "BB_Revert": lambda df: _bb_revert(df),
    "RSI_MR": lambda df: _rsi_mr(df),
    # ── ML Proxy ─────────────────────────────
    "ML_Ensemble": lambda df: _ml_ensemble(df),
    # ── Volatility ────────────────────────────
    "VRP": lambda df: _vrp(df),
    # ── Multi-Factor ─────────────────────────
    "MultiFactor": lambda df: _multifactor(df),
}


def _tsmom(df):
    c = df["close"]
    if len(c) < 252: return 0
    r = c.iloc[-1] / c.iloc[-252] - 1
    return 1 if r > 0 else -1

def _ma_cross(df):
    c = df["close"]
    if len(c) < 50: return 0
    f, s = _sma(c, 20).iloc[-1], _sma(c, 50).iloc[-1]
    fp, sp = _sma(c, 20).iloc[-2], _sma(c, 50).iloc[-2]
    if fp <= sp and f > s: return 1
    if fp >= sp and f < s: return -1
    return 0

def _reversal(df):
    c = df["close"]
    if len(c) < 10: return 0
    r5 = c.iloc[-1] / c.iloc[-5] - 1
    rsi = _rsi(c, 7).iloc[-1]
    if r5 < -0.04 and rsi < 35: return 1
    if r5 >  0.04 and rsi > 65: return -1
    return 0

def _bb_revert(df):
    c = df["close"]
    if len(c) < 22: return 0
    up, _, lo = _bb(c)
    p = c.iloc[-1]
    if p < lo.iloc[-1]: return 1
    if p > up.iloc[-1]: return -1
    return 0

def _rsi_mr(df):
    c = df["close"]
    if len(c) < 16: return 0
    r = _rsi(c).iloc[-1]
    if r < 30: return 1
    if r > 70: return -1
    return 0

def _ml_ensemble(df):
    c = df["close"]
    if len(c) < 60: return 0
    score = 0
    score += 0.25 * np.sign(c.pct_change(20).iloc[-1])
    ml, ms = _macd(c)
    score += 0.25 * np.sign((ml - ms).iloc[-1])
    score += 0.20 * (1 if _rsi(c).iloc[-1] < 45 else (-1 if _rsi(c).iloc[-1] > 55 else 0))
    up, _, lo = _bb(c)
    score += 0.15 * np.sign(lo.iloc[-1] - c.iloc[-1])
    score += 0.15 * np.sign(_sma(c, 60).iloc[-1] - c.iloc[-1])
    if score > 0.25: return 1
    if score < -0.25: return -1
    return 0

def _vrp(df):
    c = df["close"]
    if len(c) < 25: return 0
    rv  = c.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    vrp = rv * 0.15  # IV typically 15% above RV
    if vrp > 2: return -1   # sell vol (options expensive)
    if vrp < -2: return 1
    return 0

def _multifactor(df):
    c = df["close"]
    if len(c) < 60: return 0
    n = min(252, len(c)-1)
    mom   = np.sign((c.iloc[-1] / c.iloc[-n]) - 1)
    val   = np.sign(_sma(c, min(200, len(c))).iloc[-1] / c.iloc[-1] - 1)
    vol   = c.pct_change().rolling(60).std().iloc[-1] * np.sqrt(252)
    qual  = -np.sign(vol - 0.25)
    comp  = 0.4*mom + 0.3*val + 0.3*qual
    if comp > 0.25: return 1
    if comp < -0.25: return -1
    return 0


# ─────────────────────────────────────────────
# DATA FETCH (yfinance wrapper)
# ─────────────────────────────────────────────

def fetch_data(symbol: str, days: int = 400) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        from datetime import timedelta
        end   = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        if len(df) < 60:
            return None
        return df
    except Exception:
        return None


# ─────────────────────────────────────────────
# WALK-FORWARD BACKTEST ENGINE
# ─────────────────────────────────────────────

def backtest_strategy(df: pd.DataFrame, strategy_fn, hold_days: int = 5) -> dict:
    """
    Walk-forward backtest:
    - For each day in [lookback..end-hold_days], compute signal
    - Record actual return over next `hold_days`
    - Compute: total_return, sharpe, win_rate, max_drawdown
    """
    close   = df["close"].astype(float)
    signals = []
    returns = []

    step = max(1, len(close) // 60)   # ~60 test points max for speed

    for i in range(60, len(close) - hold_days, step):
        sub_df = df.iloc[:i].copy()
        try:
            sig = strategy_fn(sub_df)
        except Exception:
            sig = 0
        if sig == 0:
            continue
        future_ret = (close.iloc[i + hold_days] / close.iloc[i]) - 1
        strategy_ret = sig * future_ret       # +1 or -1 × actual return
        signals.append(sig)
        returns.append(strategy_ret)

    if len(returns) < 5:
        return {"n_trades": len(returns), "total_return": 0.0, "sharpe": 0.0,
                "win_rate": 0.0, "max_drawdown": 0.0, "avg_return": 0.0}

    r = pd.Series(returns)
    cumulative    = (1 + r).cumprod()
    total_return  = float(cumulative.iloc[-1] - 1)
    avg_ret       = float(r.mean())
    sharpe        = float(r.mean() / (r.std() + 1e-10)) * np.sqrt(252 / hold_days)
    win_rate      = float((r > 0).sum() / len(r))
    rolling_max   = cumulative.cummax()
    drawdown      = (cumulative - rolling_max) / rolling_max
    max_dd        = float(drawdown.min())

    return {
        "n_trades":     len(returns),
        "total_return": round(total_return * 100, 2),
        "sharpe":       round(sharpe, 3),
        "win_rate":     round(win_rate * 100, 1),
        "max_drawdown": round(max_dd * 100, 2),
        "avg_return":   round(avg_ret * 100, 3),
    }


# ─────────────────────────────────────────────
# CONFIDENCE SCORE
# ─────────────────────────────────────────────

def confidence_score(backtest_results: dict, current_signal: int, n_strats_agree: int) -> float:
    """
    Combine backtest quality with current signal agreement into 0–100 score.
    """
    score = 0.0

    # Backtest quality component (0–50)
    best = max(backtest_results.values(), key=lambda x: x.get("sharpe", 0))
    score += min(max(best.get("sharpe", 0) * 10, 0), 20)         # Sharpe → 0-20
    score += min(max(best.get("win_rate", 0) - 50, 0), 10)       # Win rate above 50% → 0-10
    score += min(max(best.get("total_return", 0) / 2, 0), 10)    # Total return → 0-10
    if best.get("max_drawdown", -100) > -15:
        score += 10                                                # Low drawdown bonus

    # Signal agreement component (0–50)
    score += n_strats_agree * (50 / max(len(STRATEGIES), 1))

    return round(min(score, 100), 1)


# ─────────────────────────────────────────────
# SINGLE ASSET ANALYSIS
# ─────────────────────────────────────────────

def analyze_asset(symbol: str, days: int = 365, verbose: bool = False) -> Optional[dict]:
    df = fetch_data(symbol, days + 50)
    if df is None:
        return None

    close    = df["close"].astype(float)
    price    = float(close.iloc[-1])
    vol_20   = float(close.pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) * 100

    # Run all strategies: current signal + backtest
    current_signals = {}
    bt_results      = {}
    for name, fn in STRATEGIES.items():
        current_signals[name] = fn(df)
        bt_results[name]      = backtest_strategy(df, fn)

    buy_votes  = sum(1 for v in current_signals.values() if v == 1)
    sell_votes = sum(1 for v in current_signals.values() if v == -1)

    if buy_votes > sell_votes:
        final_signal = "BUY"
        agree_count  = buy_votes
    elif sell_votes > buy_votes:
        final_signal = "SELL"
        agree_count  = sell_votes
    else:
        final_signal = "NEUTRAL"
        agree_count  = 0

    net_signal = buy_votes - sell_votes

    # Best backtest strategy
    best_strat = max(bt_results, key=lambda k: bt_results[k]["sharpe"])
    best_bt    = bt_results[best_strat]

    conf = confidence_score(bt_results, net_signal, agree_count)

    result = {
        "symbol":       symbol,
        "price":        round(price, 4),
        "vol_20d_pct":  round(vol_20, 1),
        "signal":       final_signal,
        "confidence":   conf,
        "buy_votes":    buy_votes,
        "sell_votes":   sell_votes,
        "best_strategy":best_strat,
        "bt_sharpe":    best_bt["sharpe"],
        "bt_return_pct":best_bt["total_return"],
        "bt_win_rate":  best_bt["win_rate"],
        "bt_maxdd_pct": best_bt["max_drawdown"],
        "bt_trades":    best_bt["n_trades"],
        **{f"sig_{k}": v for k, v in current_signals.items()},
    }

    if verbose:
        _print_asset_report(symbol, df, current_signals, bt_results, result)

    return result


def _print_asset_report(symbol, df, signals, bt_results, summary):
    close = df["close"].astype(float)
    print(f"\n{'═'*60}")
    print(f"  {symbol}   price=${summary['price']}   vol={summary['vol_20d_pct']}%")
    print(f"  Final signal : {summary['signal']}  (confidence={summary['confidence']:.0f}/100)")
    print(f"{'─'*60}")
    print(f"  {'Strategy':<18} {'Signal':<8} {'Sharpe':>7} {'Return%':>9} {'WinRate%':>9} {'MaxDD%':>8}")
    print(f"  {'─'*58}")
    sig_map = {1: "BUY", -1: "SELL", 0: "—"}
    for name, sig in signals.items():
        bt = bt_results[name]
        marker = " ←" if name == summary["best_strategy"] else ""
        print(f"  {name:<18} {sig_map[sig]:<8} {bt['sharpe']:>7.3f} "
              f"{bt['total_return']:>8.1f}% {bt['win_rate']:>8.1f}% "
              f"{bt['max_drawdown']:>7.1f}%{marker}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────
# FULL SCAN
# ─────────────────────────────────────────────

def run_scan(assets: list, days: int = 365) -> pd.DataFrame:
    print(f"\n{'═'*65}")
    print(f"  BACKTEST SCAN  —  {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  {len(assets)} assets × {len(STRATEGIES)} strategies")
    print(f"{'═'*65}")

    rows = []
    total = len(assets)
    for i, sym in enumerate(assets, 1):
        pct = i / total * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:5.1f}%  {sym:<12}", end="", flush=True)
        row = analyze_asset(sym, days=days)
        if row:
            rows.append(row)

    print(f"\r  {'█'*20} 100.0%  Done!{' '*20}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(df: pd.DataFrame, top_n: int = 20, signal_filter: str = None):
    if df.empty:
        print("  No results.")
        return

    # Separate buy, sell, neutral
    buys    = df[df["signal"] == "BUY"].sort_values("confidence", ascending=False)
    sells   = df[df["signal"] == "SELL"].sort_values("confidence", ascending=False)
    neutral = df[df["signal"] == "NEUTRAL"].sort_values("confidence", ascending=False)

    def _table(subset, label):
        if subset.empty:
            return
        cols = ["symbol", "price", "confidence", "buy_votes", "sell_votes",
                "bt_sharpe", "bt_return_pct", "bt_win_rate", "bt_maxdd_pct", "best_strategy"]
        sub = subset[cols].head(top_n).reset_index(drop=True)
        sub.index += 1
        print(f"\n  ── {label} ({len(subset)} total, showing top {min(top_n, len(subset))}) ──\n")
        # Pretty print
        header = (f"  {'#':>3}  {'Symbol':<12} {'Price':>10} {'Conf':>5} "
                  f"{'Buys':>5} {'Sells':>5} {'Sharpe':>7} "
                  f"{'Ret%':>7} {'Win%':>6} {'MaxDD%':>7}  Best Strategy")
        print(header)
        print("  " + "─" * (len(header) - 2))
        for rank, row in sub.iterrows():
            print(f"  {rank:>3}  {row['symbol']:<12} {row['price']:>10.4f} "
                  f"{row['confidence']:>5.0f} "
                  f"{row['buy_votes']:>5.0f} {row['sell_votes']:>5.0f} "
                  f"{row['bt_sharpe']:>7.3f} "
                  f"{row['bt_return_pct']:>6.1f}% {row['bt_win_rate']:>5.1f}% "
                  f"{row['bt_maxdd_pct']:>6.1f}%  {row['best_strategy']}")

    print(f"\n{'═'*65}")
    print(f"  RESULTS  —  {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  {len(df)} assets scanned  |  "
          f"BUY: {len(buys)}  SELL: {len(sells)}  NEUTRAL: {len(neutral)}")
    print(f"{'═'*65}")

    if signal_filter in (None, "BUY"):
        _table(buys, "BUY OPPORTUNITIES")
    if signal_filter in (None, "SELL"):
        _table(sells, "SELL OPPORTUNITIES")
    if signal_filter is None and not neutral.empty:
        print(f"\n  ── NEUTRAL — {len(neutral)} assets (not shown) ──")

    print(f"\n{'═'*65}")
    print("  Confidence score: backtest quality (50%) + strategy agreement (50%)")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────
# CHART (optional, requires matplotlib)
# ─────────────────────────────────────────────

def plot_top_opportunities(df: pd.DataFrame, n: int = 10):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gs
    except ImportError:
        print("  Install matplotlib for charts: pip install matplotlib")
        return

    actionable = df[df["signal"] != "NEUTRAL"].sort_values("confidence", ascending=False).head(n)
    if actionable.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: confidence bar chart
    colors = ["#2ecc71" if s == "BUY" else "#e74c3c" for s in actionable["signal"]]
    bars = axes[0].barh(actionable["symbol"][::-1], actionable["confidence"][::-1],
                        color=colors[::-1], alpha=0.8, edgecolor='white')
    axes[0].set_xlabel("Confidence Score (0–100)")
    axes[0].set_title(f"Top {n} Opportunities — Confidence", fontweight='bold')
    axes[0].axvline(50, color='gray', ls='--', alpha=0.5, label='50 threshold')
    for bar, (_, row) in zip(bars, actionable[::-1].iterrows()):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     row["signal"], va='center', fontsize=9, fontweight='bold',
                     color='green' if row["signal"] == "BUY" else 'red')
    axes[0].legend()

    # Right: Sharpe vs Win Rate scatter
    buy_mask  = df["signal"] == "BUY"
    sell_mask = df["signal"] == "SELL"
    axes[1].scatter(df[buy_mask]["bt_sharpe"],  df[buy_mask]["bt_win_rate"],
                    c='#2ecc71', s=60, alpha=0.7, label='BUY', zorder=3)
    axes[1].scatter(df[sell_mask]["bt_sharpe"], df[sell_mask]["bt_win_rate"],
                    c='#e74c3c', s=60, alpha=0.7, label='SELL', zorder=3)
    axes[1].scatter(df[~(buy_mask|sell_mask)]["bt_sharpe"],
                    df[~(buy_mask|sell_mask)]["bt_win_rate"],
                    c='gray', s=30, alpha=0.4, label='NEUTRAL', zorder=2)
    # Label top opportunities
    for _, row in actionable.head(5).iterrows():
        axes[1].annotate(row["symbol"], (row["bt_sharpe"], row["bt_win_rate"]),
                         fontsize=8, ha='left', xytext=(3, 3), textcoords='offset points')
    axes[1].axhline(50, color='gray', ls='--', alpha=0.4)
    axes[1].axvline(0,  color='gray', ls='--', alpha=0.4)
    axes[1].set_xlabel("Best Strategy Sharpe Ratio")
    axes[1].set_ylabel("Backtest Win Rate (%)")
    axes[1].set_title("Backtest Quality — All Assets", fontweight='bold')
    axes[1].legend()

    plt.suptitle(f"Backtest Results — {datetime.today().strftime('%Y-%m-%d')}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("  Chart saved to backtest_results.png")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backtest 100 most-traded assets and rank current opportunities"
    )
    parser.add_argument("--top",    type=int,  default=20,     help="Show top N opportunities")
    parser.add_argument("--signal", choices=["BUY", "SELL"],   help="Filter by signal type")
    parser.add_argument("--asset",  metavar="SYM",             help="Analyse a single asset in detail")
    parser.add_argument("--days",   type=int,  default=365,    help="Lookback days for backtest")
    parser.add_argument("--no-save",action="store_true",       help="Skip saving CSV")
    parser.add_argument("--no-chart",action="store_true",      help="Skip matplotlib chart")
    parser.add_argument("--assets", metavar="A,B,C",           help="Custom comma-separated symbol list")
    args = parser.parse_args()

    # ── Single asset detail mode ─────────────
    if args.asset:
        analyze_asset(args.asset.upper(), days=args.days, verbose=True)
        return

    # ── Asset list ───────────────────────────
    if args.assets:
        asset_list = [s.strip().upper() for s in args.assets.split(",")]
    else:
        asset_list = ALL_ASSETS

    # ── Full scan ────────────────────────────
    results = run_scan(asset_list, days=args.days)

    # ── Print report ─────────────────────────
    print_report(results, top_n=args.top, signal_filter=args.signal)

    # ── Save CSV ─────────────────────────────
    if not args.no_save and not results.empty:
        fname = f"backtest_results_{datetime.today().strftime('%Y%m%d')}.csv"
        results.to_csv(fname, index=False)
        print(f"  Full results saved → {fname}")

    # ── Chart ────────────────────────────────
    if not args.no_chart and not results.empty:
        plot_top_opportunities(results, n=args.top)


if __name__ == "__main__":
    main()

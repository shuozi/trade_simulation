"""
transaction.py — Transaction Cost Engine
=========================================
Models brokerage fees and integrates them into portfolio P&L.

Broker: YUH (PostFinance × Swissquote), Switzerland
Fee data crawled from yuh.com/en/fees

Features
--------
  • Per-trade cost breakdown (trading fee + FX spread)
  • Round-trip P&L: gross → net after all fees
  • Break-even price calculation
  • Cost drag as % of invested capital
  • FX rate fetching (USD/CHF, EUR/CHF, …)
  • Annotates trades.json with cost metadata
  • CLI: python transaction.py --symbol AAPL --price 200 --qty 10

Usage (import)
--------------
    from transaction import YUH, annotate_trades, portfolio_cost_summary

    cost = YUH.trade_cost("AAPL", price=200, qty=10)
    print(cost)

    pnl  = YUH.roundtrip(buy_price=200, sell_price=220, qty=10, symbol="AAPL")
    print(pnl)
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FX RATE HELPER
# ─────────────────────────────────────────────────────────────────────────────

_fx_cache: dict = {}


def get_fx_rate(from_currency: str, to_currency: str = "CHF") -> float:
    """
    Return the spot exchange rate from_currency → to_currency.
    Cached per session.  Falls back to 1.0 if unavailable.
    """
    if from_currency == to_currency:
        return 1.0
    key = f"{from_currency}{to_currency}"
    if key in _fx_cache:
        return _fx_cache[key]
    try:
        import yfinance as yf
        ticker = f"{from_currency}{to_currency}=X"
        hist   = yf.Ticker(ticker).history(period="2d")
        if hist.empty:
            _fx_cache[key] = 1.0
            return 1.0
        rate = float(hist["Close"].iloc[-1])
        _fx_cache[key] = rate
        return rate
    except Exception:
        _fx_cache[key] = 1.0
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# ASSET CLASSIFIER  (currency + type from yfinance ticker format)
# ─────────────────────────────────────────────────────────────────────────────

def classify_ticker(symbol: str) -> dict:
    """
    Derive asset type and native currency from the ticker format.

    Returns
    -------
    {type, currency, exchange, needs_fx}
    """
    s = symbol.upper()

    # Crypto  ─ ends with -USD, -BTC, -ETH, -USDT
    if any(s.endswith(sfx) for sfx in ["-USD", "-BTC", "-ETH", "-USDT"]):
        return {"type": "crypto",  "currency": "USD", "exchange": "Crypto",        "needs_fx": True}

    # Swiss exchange (.SW)
    if s.endswith(".SW"):
        return {"type": "equity",  "currency": "CHF", "exchange": "SIX",           "needs_fx": False}

    # London (.L)
    if s.endswith(".L"):
        return {"type": "equity",  "currency": "GBP", "exchange": "LSE",           "needs_fx": True}

    # German XETRA (.DE or .F)
    if s.endswith(".DE") or s.endswith(".F"):
        return {"type": "equity",  "currency": "EUR", "exchange": "XETRA",         "needs_fx": True}

    # Euronext Paris / Amsterdam / Brussels
    if s.endswith(".PA") or s.endswith(".AS") or s.endswith(".BR"):
        return {"type": "equity",  "currency": "EUR", "exchange": "Euronext",      "needs_fx": True}

    # Tokyo
    if s.endswith(".T"):
        return {"type": "equity",  "currency": "JPY", "exchange": "TSE",           "needs_fx": True}

    # Hong Kong
    if s.endswith(".HK"):
        return {"type": "equity",  "currency": "HKD", "exchange": "HKEX",          "needs_fx": True}

    # Australia
    if s.endswith(".AX"):
        return {"type": "equity",  "currency": "AUD", "exchange": "ASX",           "needs_fx": True}

    # Canada
    if s.endswith(".TO"):
        return {"type": "equity",  "currency": "CAD", "exchange": "TSX",           "needs_fx": True}

    # Known ETFs (US-listed)
    _ETF = {"SPY","QQQ","IWM","DIA","GLD","SLV","TLT","HYG","LQD","VXX",
            "ARKK","XLK","XLF","XLE","XLV","XLI","XLP","XLU","XLY","XLB",
            "IAU","VTI","VOO","VEA","VWO","IEFA","EEM","EFA","AGG","BND",
            "SCHD","JEPI","QYLD","XYLD","RSP","MDY","IJR","IVV","ITOT"}
    if s in _ETF:
        return {"type": "etf",    "currency": "USD", "exchange": "US",            "needs_fx": True}

    # Default: US equity (NYSE/NASDAQ)
    return     {"type": "equity",  "currency": "USD", "exchange": "US",            "needs_fx": True}


# ─────────────────────────────────────────────────────────────────────────────
# BROKER COST MODEL  (base class)
# ─────────────────────────────────────────────────────────────────────────────

class BrokerCostModel:
    name        = "Generic Broker"
    base_currency = "CHF"

    # Override in subclasses
    EQUITY_FEE_PCT  = 0.005      # 0.50 %
    CRYPTO_FEE_PCT  = 0.010      # 1.00 %
    ETF_FEE_PCT     = 0.005      # 0.50 %
    ETF_RECURRING_FEE_PCT = 0.0  # free for eligible recurring ETF buys
    FX_SPREAD_PCT   = 0.0        # currency conversion markup
    MIN_FEE         = 0.0        # minimum fee in base currency
    CUSTODY_FEE_PCT = 0.0        # annual custody fee
    NOTE            = ""

    # ── Core cost computation ─────────────────────────────────────────────

    def _fee_pct(self, asset_type: str, recurring: bool = False) -> float:
        if asset_type == "crypto":
            return self.CRYPTO_FEE_PCT
        if asset_type == "etf":
            return self.ETF_RECURRING_FEE_PCT if recurring else self.ETF_FEE_PCT
        return self.EQUITY_FEE_PCT

    def trade_cost(self, symbol: str, price: float, qty: float,
                   side: str = "BUY", recurring: bool = False,
                   fx_rate_to_base: Optional[float] = None) -> dict:
        """
        Compute the full cost of a single trade in base currency (CHF).

        Returns
        -------
        dict with keys:
            symbol, side, price, qty, asset_type, currency,
            gross_value_native, gross_value_chf,
            trading_fee_native, trading_fee_chf,
            fx_fee_chf,
            total_cost_chf,
            effective_fee_pct,
            cost_per_unit
        """
        info        = classify_ticker(symbol)
        asset_type  = info["type"]
        currency    = info["currency"]
        needs_fx    = info["needs_fx"]

        gross_native = price * qty                     # e.g. USD 2 000

        # FX rate to CHF
        if fx_rate_to_base is None:
            fx_rate = get_fx_rate(currency, self.base_currency) if needs_fx else 1.0
        else:
            fx_rate  = fx_rate_to_base

        gross_chf    = gross_native * fx_rate          # e.g. 1 800 CHF

        # Trading fee
        fee_pct      = self._fee_pct(asset_type, recurring)
        fee_native   = max(gross_native * fee_pct, self.MIN_FEE / fx_rate if fx_rate else 0)
        fee_chf      = fee_native * fx_rate

        # FX conversion cost (only if non-CHF)
        fx_fee_chf   = gross_chf * self.FX_SPREAD_PCT if needs_fx else 0.0

        total_chf    = fee_chf + fx_fee_chf
        eff_pct      = (total_chf / gross_chf * 100) if gross_chf else 0.0

        return {
            "symbol":              symbol,
            "side":                side,
            "price":               round(price, 4),
            "qty":                 round(qty,   4),
            "asset_type":          asset_type,
            "currency":            currency,
            "exchange":            info["exchange"],
            "fx_rate":             round(fx_rate, 4),
            "gross_value_native":  round(gross_native, 2),
            "gross_value_chf":     round(gross_chf,    2),
            "trading_fee_native":  round(fee_native,   4),
            "trading_fee_chf":     round(fee_chf,      2),
            "fx_fee_chf":          round(fx_fee_chf,   2),
            "total_cost_chf":      round(total_chf,    2),
            "effective_fee_pct":   round(eff_pct,      3),
            "cost_per_unit":       round(total_chf / qty if qty else 0, 4),
            "recurring":           recurring,
            "broker":              self.name,
        }

    def roundtrip(self, buy_price: float, sell_price: float,
                  qty: float, symbol: str,
                  buy_fx: Optional[float] = None,
                  sell_fx: Optional[float] = None) -> dict:
        """
        Full round-trip P&L: gross → net after all buy + sell costs.

        Returns
        -------
        dict with breakeven price, gross/net P&L, cost drag %
        """
        buy_info  = self.trade_cost(symbol, buy_price,  qty, "BUY",  fx_rate_to_base=buy_fx)
        sell_info = self.trade_cost(symbol, sell_price, qty, "SELL", fx_rate_to_base=sell_fx)

        info     = classify_ticker(symbol)
        currency = info["currency"]
        fx       = buy_info["fx_rate"]

        gross_pnl_native = (sell_price - buy_price) * qty
        gross_pnl_chf    = gross_pnl_native * fx

        total_cost_chf   = buy_info["total_cost_chf"] + sell_info["total_cost_chf"]
        net_pnl_chf      = gross_pnl_chf - total_cost_chf

        invested_chf     = buy_info["gross_value_chf"]
        gross_ret_pct    = (gross_pnl_chf / invested_chf * 100) if invested_chf else 0
        net_ret_pct      = (net_pnl_chf   / invested_chf * 100) if invested_chf else 0
        cost_drag_pct    = gross_ret_pct - net_ret_pct

        # Break-even sell price (net P&L = 0)
        # We need: (breakeven - buy_price)*qty*fx = total_cost_chf
        breakeven_native = buy_price + (total_cost_chf / fx / qty) if (fx * qty) else buy_price
        breakeven_pct    = ((breakeven_native / buy_price) - 1) * 100 if buy_price else 0

        return {
            "symbol":             symbol,
            "buy_price":          round(buy_price,       4),
            "sell_price":         round(sell_price,      4),
            "qty":                round(qty,              4),
            "currency":           currency,
            "fx_rate":            round(fx,               4),
            "gross_pnl_native":   round(gross_pnl_native, 2),
            "gross_pnl_chf":      round(gross_pnl_chf,    2),
            "buy_cost_chf":       round(buy_info["total_cost_chf"],  2),
            "sell_cost_chf":      round(sell_info["total_cost_chf"], 2),
            "total_cost_chf":     round(total_cost_chf,   2),
            "net_pnl_chf":        round(net_pnl_chf,      2),
            "gross_return_pct":   round(gross_ret_pct,    3),
            "net_return_pct":     round(net_ret_pct,      3),
            "cost_drag_pct":      round(cost_drag_pct,    3),
            "breakeven_price":    round(breakeven_native, 4),
            "breakeven_pct":      round(breakeven_pct,    3),
            "broker":             self.name,
        }

    def annual_custody(self, portfolio_value_chf: float) -> float:
        """Annual custody fee in CHF."""
        return round(portfolio_value_chf * self.CUSTODY_FEE_PCT, 2)


# ─────────────────────────────────────────────────────────────────────────────
# YUH BROKER  (PostFinance × Swissquote, Switzerland)
# Fee source: yuh.com/en/fees  —  crawled 2026-04-03
# ─────────────────────────────────────────────────────────────────────────────

class _YUHCosts(BrokerCostModel):
    """
    YUH Switzerland fee schedule (as of 2026-04-03).

    Stocks & ETFs  : 0.50% per trade
    Crypto         : 1.00% per trade
    Recurring ETFs : FREE (eligible YUH Select ETFs only)
    FX spread      : 0.95% (inter-bank rate + markup)
    Custody        : FREE
    Minimum fee    : none documented
    """
    name              = "YUH (PostFinance × Swissquote)"
    base_currency     = "CHF"

    EQUITY_FEE_PCT       = 0.0050   # 0.50 %
    CRYPTO_FEE_PCT       = 0.0100   # 1.00 %
    ETF_FEE_PCT          = 0.0050   # 0.50 %
    ETF_RECURRING_FEE_PCT= 0.0000   # free for eligible recurring buys
    FX_SPREAD_PCT        = 0.0095   # 0.95 % currency conversion markup
    MIN_FEE              = 0.0      # no documented minimum
    CUSTODY_FEE_PCT      = 0.0      # no custody fee

    NOTE = (
        "YUH pricing as of 2026-04-03 from yuh.com/en/fees:\n"
        "  • Stocks / ETFs  : 0.50 % per trade\n"
        "  • Crypto         : 1.00 % per trade\n"
        "  • Recurring ETFs : FREE (select eligible ETFs)\n"
        "  • FX spread      : 0.95 % (inter-bank rate + markup)\n"
        "  • Custody        : FREE\n"
        "  • ATM CH         : CHF 1.90 (1 free/week)\n"
        "  • ATM abroad     : CHF 4.90\n"
        "  • SEPA EUR       : FREE\n"
        "  • 3a pension     : 0.50 % all-in fee"
    )

YUH = _YUHCosts()


# ─────────────────────────────────────────────────────────────────────────────
# TRADE ANNOTATION  (integrate with trades.json)
# ─────────────────────────────────────────────────────────────────────────────

def annotate_trades(trades: list, broker: BrokerCostModel = YUH) -> list:
    """
    Add cost metadata to each trade dict (non-destructive — adds _cost key).
    Compatible with trades.json format used by exercise.py.
    """
    annotated = []
    for t in trades:
        t = dict(t)
        try:
            cost = broker.trade_cost(
                symbol = t["symbol"],
                price  = float(t["price"]),
                qty    = float(t["quantity"]),
                side   = t.get("side", "BUY"),
            )
            t["_cost"] = cost
        except Exception as e:
            t["_cost"] = {"error": str(e), "total_cost_chf": 0}
        annotated.append(t)
    return annotated


def portfolio_cost_summary(trades: list, broker: BrokerCostModel = YUH) -> dict:
    """
    Compute total fees paid across all trades.

    Returns
    -------
    {
        total_trading_fees_chf,
        total_fx_fees_chf,
        total_costs_chf,
        by_symbol: {sym: {trades, total_cost_chf}},
        most_expensive_trade,
        avg_cost_per_trade_chf,
    }
    """
    annotated = annotate_trades(trades, broker)
    total_trade = 0.0
    total_fx    = 0.0
    by_symbol: dict = {}
    most_exp    = None
    most_exp_cost = -1.0

    for t in annotated:
        c = t.get("_cost", {})
        if "error" in c:
            continue
        tc  = c.get("total_cost_chf",  0)
        fx  = c.get("fx_fee_chf",       0)
        trd = c.get("trading_fee_chf",  0)
        total_trade += trd
        total_fx    += fx

        sym = t["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "total_cost_chf": 0.0}
        by_symbol[sym]["trades"]          += 1
        by_symbol[sym]["total_cost_chf"]  += tc

        if tc > most_exp_cost:
            most_exp_cost = tc
            most_exp      = {**t, "_cost_chf": tc}

    n = len([t for t in annotated if "error" not in t.get("_cost", {})])
    return {
        "total_trading_fees_chf":  round(total_trade, 2),
        "total_fx_fees_chf":       round(total_fx,    2),
        "total_costs_chf":         round(total_trade + total_fx, 2),
        "by_symbol":               {k: {**v, "total_cost_chf": round(v["total_cost_chf"], 2)}
                                    for k, v in by_symbol.items()},
        "most_expensive_trade":    most_exp,
        "avg_cost_per_trade_chf":  round((total_trade + total_fx) / n, 2) if n else 0,
        "n_trades":                n,
        "broker":                  broker.name,
    }


def net_pnl_after_costs(buy_trade: dict, sell_trade: dict,
                        broker: BrokerCostModel = YUH) -> dict:
    """
    Given a matched buy→sell pair (from FIFO matching), return net P&L
    after accounting for brokerage costs.

    Parameters
    ----------
    buy_trade, sell_trade : dicts with keys symbol, price, quantity
    """
    sym = buy_trade["symbol"]
    return broker.roundtrip(
        buy_price  = float(buy_trade["price"]),
        sell_price = float(sell_trade["price"]),
        qty        = float(buy_trade["quantity"]),
        symbol     = sym,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY COST IMPACT  (interaction with strategy signals)
# ─────────────────────────────────────────────────────────────────────────────

def cost_adjusted_signal(signal_score: int, symbol: str, price: float,
                         qty: float = 1.0,
                         broker: BrokerCostModel = YUH,
                         min_expected_return_pct: float = 2.0) -> dict:
    """
    Adjust a raw strategy signal score by whether the expected return
    justifies the round-trip transaction cost.

    A BUY signal is only worth acting on if the strategy's minimum
    expected return exceeds the round-trip cost threshold.

    Parameters
    ----------
    signal_score           : raw signal score (-8 to +8)
    min_expected_return_pct: minimum expected return for this signal strength

    Returns
    -------
    {
        original_score, adjusted_score, cost_drag_pct,
        breakeven_pct, worthwhile, reason
    }
    """
    cost_info = broker.trade_cost(symbol, price, qty)
    # Round-trip drag = 2× one-way cost (buy + sell)
    roundtrip_drag = cost_info["effective_fee_pct"] * 2

    info        = classify_ticker(symbol)
    fx_drag     = (broker.FX_SPREAD_PCT * 2 * 100) if info["needs_fx"] else 0
    total_drag  = roundtrip_drag + fx_drag

    # If expected return < total drag, the signal has no edge after costs
    worthwhile  = min_expected_return_pct > total_drag
    adjusted    = signal_score if worthwhile else max(signal_score - 1, -8)

    if worthwhile:
        reason = (f"Signal viable: expected {min_expected_return_pct:.1f}% > "
                  f"cost drag {total_drag:.2f}%")
    else:
        reason = (f"Signal marginal: cost drag {total_drag:.2f}% may erode "
                  f"expected return {min_expected_return_pct:.1f}%. "
                  f"Score penalised by 1.")

    return {
        "symbol":               symbol,
        "original_score":       signal_score,
        "adjusted_score":       adjusted,
        "one_way_cost_pct":     round(cost_info["effective_fee_pct"], 3),
        "roundtrip_drag_pct":   round(total_drag, 3),
        "breakeven_pct":        round(total_drag, 3),
        "worthwhile":           worthwhile,
        "reason":               reason,
        "broker":               broker.name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT HELPERS  (for integration with report.py / exercise.py)
# ─────────────────────────────────────────────────────────────────────────────

def cost_summary_cards(trades: list, broker: BrokerCostModel = YUH) -> list:
    """
    Return a list of dicts suitable for rendering as summary cards.
    Used by exercise.py dashboard.
    """
    summary = portfolio_cost_summary(trades, broker)
    return [
        {"label": "Total Fees Paid",    "value": f"CHF {summary['total_costs_chf']:,.2f}",
         "sub": f"{summary['n_trades']} trades",             "color": "#e74c3c"},
        {"label": "Trading Fees",       "value": f"CHF {summary['total_trading_fees_chf']:,.2f}",
         "sub": f"{broker.EQUITY_FEE_PCT*100:.2f}% per trade","color": "#e67e22"},
        {"label": "FX Conversion Fees", "value": f"CHF {summary['total_fx_fees_chf']:,.2f}",
         "sub": f"{broker.FX_SPREAD_PCT*100:.2f}% spread",   "color": "#f39c12"},
        {"label": "Avg Cost/Trade",     "value": f"CHF {summary['avg_cost_per_trade_chf']:,.2f}",
         "sub": "per single leg",                             "color": "#95a5a6"},
    ]


def breakeven_table(open_positions: dict, broker: BrokerCostModel = YUH) -> list:
    """
    For each open position, compute how much price must rise to break even
    after buying cost + anticipated selling cost.
    Returns list of dicts for table rendering.
    """
    rows = []
    for sym, pos in open_positions.items():
        cost = broker.trade_cost(sym, pos["avg_cost"], pos["quantity"], "BUY")
        # Breakeven: buy_cost + sell_cost = (breakeven - avg_cost) * qty * fx
        fx          = cost["fx_rate"]
        qty         = pos["quantity"]
        avg         = pos["avg_cost"]
        total_drag  = cost["total_cost_chf"] * 2          # buy + sell
        be_price    = avg + (total_drag / (fx * qty)) if (fx * qty) else avg
        be_pct      = (be_price / avg - 1) * 100 if avg else 0
        current     = pos.get("current_price", avg)
        gap_to_be   = ((current / be_price) - 1) * 100 if be_price else 0

        rows.append({
            "symbol":          sym,
            "avg_cost":        round(avg,      4),
            "current_price":   round(current,  4),
            "breakeven_price": round(be_price, 4),
            "breakeven_pct":   round(be_pct,   3),
            "gap_to_breakeven":round(gap_to_be,2),
            "roundtrip_cost":  round(total_drag, 2),
            "currency":        cost["currency"],
        })
    return sorted(rows, key=lambda r: r["gap_to_breakeven"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def cli_main():
    parser = argparse.ArgumentParser(
        description="YUH transaction cost calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python transaction.py --symbol SREN.SW --price 130 --qty 8
  python transaction.py --symbol AAPL    --price 200 --qty 10
  python transaction.py --symbol BTC-USD --price 65000 --qty 0.1
  python transaction.py --info
  python transaction.py --trades trades.json
""")
    parser.add_argument("--symbol",  help="Ticker symbol")
    parser.add_argument("--price",   type=float, help="Trade price")
    parser.add_argument("--qty",     type=float, help="Quantity")
    parser.add_argument("--sell",    type=float, help="Sell price (for round-trip P&L)")
    parser.add_argument("--trades",  help="Path to trades.json — show portfolio cost summary")
    parser.add_argument("--info",    action="store_true", help="Show YUH fee schedule")
    args = parser.parse_args()

    print(f"\n  Broker: {YUH.name}")

    if args.info:
        _print_section("YUH Fee Schedule")
        print(YUH.NOTE)
        return

    if args.trades:
        path = args.trades
        if not os.path.exists(path):
            print(f"  ✗  File not found: {path}")
            return
        trades = json.load(open(path))
        _print_section(f"Portfolio Cost Summary  ({len(trades)} trades)")
        summary = portfolio_cost_summary(trades)
        print(f"  Total costs paid   : CHF {summary['total_costs_chf']:>8.2f}")
        print(f"  └─ Trading fees    : CHF {summary['total_trading_fees_chf']:>8.2f}")
        print(f"  └─ FX fees         : CHF {summary['total_fx_fees_chf']:>8.2f}")
        print(f"  Avg cost / trade   : CHF {summary['avg_cost_per_trade_chf']:>8.2f}")
        print(f"\n  Per-Symbol Breakdown:")
        for sym, d in summary["by_symbol"].items():
            print(f"    {sym:<12}  {d['trades']} trades   CHF {d['total_cost_chf']:,.2f}")
        return

    if not args.symbol or not args.price or not args.qty:
        parser.print_help()
        return

    # Single trade cost
    _print_section(f"Trade Cost — {args.symbol}")
    c = YUH.trade_cost(args.symbol, args.price, args.qty)
    info = classify_ticker(args.symbol)
    print(f"  Asset type         : {c['asset_type'].upper()}")
    print(f"  Exchange           : {c['exchange']}")
    print(f"  Native currency    : {c['currency']}")
    print(f"  FX rate → CHF      : {c['fx_rate']:.4f}")
    print(f"  Gross value        : {c['currency']} {c['gross_value_native']:>10,.2f}")
    print(f"  Gross value CHF    : CHF {c['gross_value_chf']:>10,.2f}")
    print(f"  Trading fee        : CHF {c['trading_fee_chf']:>10,.2f}  ({YUH.EQUITY_FEE_PCT*100:.2f}%)")
    if info["needs_fx"]:
        print(f"  FX spread (0.95%)  : CHF {c['fx_fee_chf']:>10,.2f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total cost         : CHF {c['total_cost_chf']:>10,.2f}")
    print(f"  Effective fee      :     {c['effective_fee_pct']:>10.3f}%")
    print(f"  Cost per unit      : CHF {c['cost_per_unit']:>10.4f}")

    # Round-trip if --sell given
    if args.sell:
        _print_section(f"Round-Trip P&L — {args.symbol}")
        rt = YUH.roundtrip(args.price, args.sell, args.qty, args.symbol)
        direction = "▲" if rt["net_pnl_chf"] >= 0 else "▼"
        print(f"  Buy price          : {c['currency']} {rt['buy_price']:>10,.4f}")
        print(f"  Sell price         : {c['currency']} {rt['sell_price']:>10,.4f}")
        print(f"  Quantity           :     {rt['qty']}")
        print(f"  Gross P&L (CHF)    : CHF {rt['gross_pnl_chf']:>+10,.2f}  ({rt['gross_return_pct']:+.2f}%)")
        print(f"  Buy cost           : CHF {rt['buy_cost_chf']:>10,.2f}")
        print(f"  Sell cost          : CHF {rt['sell_cost_chf']:>10,.2f}")
        print(f"  Total costs        : CHF {rt['total_cost_chf']:>10,.2f}")
        print(f"  ─────────────────────────────────────────")
        print(f"  Net P&L (CHF)      : CHF {rt['net_pnl_chf']:>+10,.2f}  ({rt['net_return_pct']:+.2f}%)  {direction}")
        print(f"  Cost drag          :     {rt['cost_drag_pct']:>10.3f}%")
        print(f"  Break-even price   : {c['currency']} {rt['breakeven_price']:>10,.4f}  (+{rt['breakeven_pct']:.3f}%)")

    # Cost-adjusted signal hint
    _print_section("Signal Cost Threshold")
    adj = cost_adjusted_signal(3, args.symbol, args.price, args.qty,
                               min_expected_return_pct=2.0)
    print(f"  Round-trip cost drag: {adj['roundtrip_drag_pct']:.3f}%")
    print(f"  Min return needed   : {adj['roundtrip_drag_pct']:.3f}% to break even")
    print(f"  Signal viable?      : {'✓ YES' if adj['worthwhile'] else '✗ MARGINAL'}")
    print(f"  Note: {adj['reason']}")

    print()


if __name__ == "__main__":
    cli_main()

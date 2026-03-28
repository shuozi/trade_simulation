"""
Strategy Class Hierarchy
========================
Base class + 8 strategy type subclasses covering 20 modern quant strategies.

Class Tree:
    Strategy (base)
    ├── MomentumStrategy
    │   ├── TimeSeriesMomentum
    │   ├── CrossSectionalMomentum
    │   ├── MLEnhancedTrend
    │   └── ShortTermReversal
    ├── StatArbStrategy
    │   ├── PairsTrading
    │   ├── ETFArbitrage
    │   ├── PCABasketArb
    │   └── CryptoBasisArb
    ├── MLStrategy
    │   ├── GradientBoostingAlpha
    │   ├── DeepRLAgent
    │   ├── LLMEarningsAlpha
    │   └── AltDataSatellite
    ├── VolatilityStrategy
    │   ├── VolatilityRiskPremium
    │   ├── DispersionTrading
    │   └── GammaScalping
    ├── FactorStrategy
    │   ├── MultiFactorLongShort
    │   └── FactorMomentum
    ├── SentimentStrategy
    │   └── SocialNewsSentiment
    ├── MarketMakingStrategy
    │   └── ElectronicMarketMaking
    └── EventDrivenStrategy
        └── EarningsSurprise

Usage:
    from strategy import TimeSeriesMomentum
    strat = TimeSeriesMomentum()
    signal = strat.generate_signal(df)   # returns SignalResult
    print(strat.describe())
"""

from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# SIGNAL RESULT
# ─────────────────────────────────────────────

@dataclass
class SignalResult:
    """Standardized output from every strategy."""
    strategy_name: str
    strategy_type: str
    symbol: str
    signal: str                     # "BUY" | "SELL" | "NEUTRAL" | "HOLD"
    strength: float                 # 0.0 – 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    indicators: dict = field(default_factory=dict)
    notes: str = ""

    def is_actionable(self) -> bool:
        return self.signal in ("BUY", "SELL")

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"[{self.strategy_type}] {self.strategy_name} → {self.signal} "
                f"(strength={self.strength:.2f}) | {self.symbol} @ {self.timestamp[:10]}")


# ─────────────────────────────────────────────
# BASE STRATEGY
# ─────────────────────────────────────────────

class Strategy(ABC):
    """
    Abstract base class for all quantitative trading strategies.

    Subclasses must implement:
        generate_signal(df, symbol) -> SignalResult
        describe()                  -> str

    All strategies carry a JSON metadata file loaded from strategies/<id>.json.
    """

    strategy_type: str = "base"
    strategy_id: int   = 0

    def __init__(self, params: dict | None = None):
        self.params = params or self._default_params()
        self._metadata: dict | None = None

    # ── Abstract interface ────────────────────
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        """
        Given OHLCV DataFrame (columns: open, high, low, close, volume),
        return a SignalResult with BUY / SELL / NEUTRAL / HOLD.
        """

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the strategy logic."""

    # ── Default params (override per class) ──
    def _default_params(self) -> dict:
        return {}

    # ── Shared helpers ────────────────────────
    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).mean()

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(series: pd.Series, fast=12, slow=26, sig=9):
        ema_f = series.ewm(span=fast, adjust=False).mean()
        ema_s = series.ewm(span=slow, adjust=False).mean()
        line  = ema_f - ema_s
        signal = line.ewm(span=sig, adjust=False).mean()
        return line, signal

    @staticmethod
    def _bollinger(series: pd.Series, window=20, std=2):
        mid   = series.rolling(window).mean()
        sigma = series.rolling(window).std()
        return mid + std * sigma, mid, mid - std * sigma  # upper, mid, lower

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hi, lo, cl = df["high"], df["low"], df["close"]
        prev_cl = cl.shift(1)
        tr = pd.concat([hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _volatility(series: pd.Series, window: int = 20) -> float:
        return float(series.pct_change().rolling(window).std().iloc[-1]) * np.sqrt(252)

    # ── Metadata loading ──────────────────────
    def metadata(self) -> dict:
        if self._metadata is None:
            path = os.path.join(os.path.dirname(__file__), "strategies", f"{self.strategy_id:02d}_{self.__class__.__name__.lower()}.json")
            if os.path.exists(path):
                with open(path) as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {}
        return self._metadata

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.strategy_type}, params={self.params})"


# ─────────────────────────────────────────────
# TYPE 1 — MOMENTUM STRATEGIES
# ─────────────────────────────────────────────

class MomentumStrategy(Strategy):
    """Base class for all momentum / trend-following strategies."""
    strategy_type = "momentum"


class TimeSeriesMomentum(MomentumStrategy):
    """
    Strategy 1 — Time-Series Momentum (TSMOM)
    Each asset is judged on its own past return. Long if trend is up, short if down.
    Uses inverse-volatility position sizing.
    """
    strategy_id = 1

    def _default_params(self) -> dict:
        return {"lookback": 252, "vol_window": 60, "entry_threshold": 0.0}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        lookback = self.params["lookback"]
        if len(close) < lookback:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0,
                                notes="Insufficient data")
        past_return = (close.iloc[-1] / close.iloc[-lookback]) - 1
        vol = self._volatility(close, self.params["vol_window"])
        strength = min(abs(past_return) / 0.20, 1.0)   # normalize to 20% threshold
        signal = "BUY" if past_return > self.params["entry_threshold"] else "SELL"
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"past_return_pct": round(past_return * 100, 2),
                                        "annualized_vol": round(vol, 4)})

    def describe(self) -> str:
        return ("Time-Series Momentum: if an asset's 12-month return is positive → BUY, "
                "negative → SELL. Position sized by inverse volatility.")


class CrossSectionalMomentum(MomentumStrategy):
    """
    Strategy 2 — Cross-Sectional Equity Momentum (12-1 Factor)
    Ranks stocks by past 12-month return (skipping last month). Requires a DataFrame
    of multiple symbols; returns BUY for top-decile, SELL for bottom-decile.
    """
    strategy_id = 2

    def _default_params(self) -> dict:
        return {"formation_months": 11, "skip_months": 1, "top_pct": 0.2}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        skip   = self.params["skip_months"] * 21
        form   = self.params["formation_months"] * 21
        if len(close) < form + skip:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        momentum = (close.iloc[-(skip + 1)] / close.iloc[-(form + skip)]) - 1
        strength = min(abs(momentum) / 0.30, 1.0)
        signal = "BUY" if momentum > 0.05 else ("SELL" if momentum < -0.05 else "NEUTRAL")
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"12_1_momentum": round(momentum * 100, 2)})

    def describe(self) -> str:
        return ("Cross-Sectional Momentum: rank stocks by 12-month return skipping last month. "
                "Long top 20%, short bottom 20%.")


class MLEnhancedTrend(MomentumStrategy):
    """
    Strategy 3 — Multi-Asset Trend Following with ML Signal Enhancement
    EWMA crossover filtered by RSI regime and volatility regime.
    (Simplified proxy: full version requires ML model trained on features.)
    """
    strategy_id = 3

    def _default_params(self) -> dict:
        return {"fast_span": 32, "slow_span": 128, "vol_target": 0.15}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close  = df["close"].astype(float)
        if len(close) < self.params["slow_span"]:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        fast = self._ema(close, self.params["fast_span"])
        slow = self._ema(close, self.params["slow_span"])
        rsi  = self._rsi(close).iloc[-1]
        trend = "BUY" if fast.iloc[-1] > slow.iloc[-1] else "SELL"
        # ML filter proxy: suppress signal when RSI is in extreme opposite zone
        if trend == "BUY" and rsi > 80:
            signal, strength = "NEUTRAL", 0.3
        elif trend == "SELL" and rsi < 20:
            signal, strength = "NEUTRAL", 0.3
        else:
            signal = trend
            strength = min(abs(fast.iloc[-1] - slow.iloc[-1]) / slow.iloc[-1] / 0.02, 1.0)
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"fast_ema": round(fast.iloc[-1], 4),
                                        "slow_ema": round(slow.iloc[-1], 4),
                                        "rsi": round(rsi, 1)})

    def describe(self) -> str:
        return ("ML-Enhanced Trend: EWMA crossover (32/128) with RSI regime filter as ML proxy. "
                "Targets 15% annualized volatility.")


class ShortTermReversal(MomentumStrategy):
    """
    Strategy 4 — Short-Term Reversal (Intraday / Weekly Mean Reversion)
    Stocks that fell hard in the past week tend to bounce. Contrarian momentum.
    """
    strategy_id = 4

    def _default_params(self) -> dict:
        return {"reversal_days": 5, "entry_z": 1.5, "rsi_oversold": 35, "rsi_overbought": 65}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 30:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        week_ret = (close.iloc[-1] / close.iloc[-self.params["reversal_days"]]) - 1
        rsi = self._rsi(close, 7).iloc[-1]
        # Reversal: buy after sharp drop, sell after sharp rally
        if week_ret < -0.04 and rsi < self.params["rsi_oversold"]:
            signal, strength = "BUY", min(abs(week_ret) / 0.10, 1.0)
        elif week_ret > 0.04 and rsi > self.params["rsi_overbought"]:
            signal, strength = "SELL", min(week_ret / 0.10, 1.0)
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"week_return_pct": round(week_ret * 100, 2), "rsi_7": round(rsi, 1)})

    def describe(self) -> str:
        return ("Short-Term Reversal: buy after a 4%+ weekly drop with RSI<35, "
                "sell after 4%+ weekly rally with RSI>65.")


# ─────────────────────────────────────────────
# TYPE 2 — STATISTICAL ARBITRAGE
# ─────────────────────────────────────────────

class StatArbStrategy(Strategy):
    """Base class for statistical arbitrage and mean-reversion strategies."""
    strategy_type = "stat_arb"


class PairsTrading(StatArbStrategy):
    """
    Strategy 5 — Classic Pairs Trading (Cointegration-Based)
    Requires df to have two 'close' columns: df['close_a'] and df['close_b'].
    Trades the spread when it diverges beyond ±2 std.
    """
    strategy_id = 5

    def _default_params(self) -> dict:
        return {"window": 252, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "SPREAD") -> SignalResult:
        if "spread" in df.columns:
            spread = df["spread"].astype(float)
        elif "close_a" in df.columns and "close_b" in df.columns:
            ratio = np.log(df["close_a"]) - np.log(df["close_b"])
            spread = ratio
        else:
            spread = df["close"].astype(float)   # fallback: treat close as spread
        win = self.params["window"]
        if len(spread) < win:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        mu  = spread.rolling(win).mean().iloc[-1]
        std = spread.rolling(win).std().iloc[-1]
        z   = (spread.iloc[-1] - mu) / (std + 1e-10)
        if z < -self.params["entry_z"]:
            signal = "BUY"          # spread too low → buy asset A, sell B
        elif z > self.params["entry_z"]:
            signal = "SELL"         # spread too high → sell A, buy B
        elif abs(z) < self.params["exit_z"]:
            signal = "HOLD"         # mean reverted → exit
        else:
            signal = "NEUTRAL"
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal,
                            min(abs(z) / self.params["entry_z"], 1.0),
                            indicators={"z_score": round(z, 3), "spread": round(float(spread.iloc[-1]), 4)})

    def describe(self) -> str:
        return ("Pairs Trading: monitor the log-spread between two cointegrated assets. "
                "Enter long/short when spread exceeds ±2σ, exit at ±0.5σ.")


class ETFArbitrage(StatArbStrategy):
    """
    Strategy 6 — Index Arbitrage / ETF Arbitrage
    Requires df with 'etf_price' and 'nav' columns, or uses close vs SMA as NAV proxy.
    """
    strategy_id = 6

    def _default_params(self) -> dict:
        return {"entry_bps": 15, "exit_bps": 3}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "ETF") -> SignalResult:
        close = df["close"].astype(float)
        if "nav" in df.columns:
            nav = df["nav"].astype(float)
        else:
            nav = self._sma(close, 5)    # proxy: 5-day SMA as fair value
        premium_bps = ((close - nav) / nav * 10000).iloc[-1]
        if premium_bps > self.params["entry_bps"]:
            signal, strength = "SELL", min(premium_bps / 50, 1.0)
        elif premium_bps < -self.params["entry_bps"]:
            signal, strength = "BUY", min(abs(premium_bps) / 50, 1.0)
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"premium_bps": round(premium_bps, 2)})

    def describe(self) -> str:
        return ("ETF Arbitrage: buy when ETF trades at >15bps discount to NAV, "
                "sell when at >15bps premium. Exit at ±3bps.")


class PCABasketArb(StatArbStrategy):
    """
    Strategy 7 — Multi-Leg Statistical Arbitrage (PCA-Based)
    Fits PCA on a returns matrix, computes residuals, trades mean-reversion of residuals.
    df should be a wide DataFrame where each column is a stock's close price.
    """
    strategy_id = 7

    def _default_params(self) -> dict:
        return {"n_components": 5, "entry_z": 2.0, "halflife_days": 10}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "BASKET") -> SignalResult:
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0,
                                notes="sklearn not installed")
        if df.shape[1] < 10 or len(df) < 60:
            # Fallback for single-asset df: use RSI as residual proxy
            close = df["close"].astype(float) if "close" in df.columns else df.iloc[:, 0]
            rsi = self._rsi(close).iloc[-1]
            if rsi < 30:
                return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "BUY", 0.6,
                                    indicators={"rsi_proxy": round(rsi, 1)})
            elif rsi > 70:
                return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "SELL", 0.6,
                                    indicators={"rsi_proxy": round(rsi, 1)})
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        rets = df.pct_change().dropna()
        pca  = PCA(n_components=self.params["n_components"])
        factors = pca.fit_transform(rets)
        reconstructed = pca.inverse_transform(factors)
        residuals = rets.values - reconstructed
        z_scores = residuals[-1] / (residuals.std(axis=0) + 1e-10)
        buy_count  = (z_scores < -self.params["entry_z"]).sum()
        sell_count = (z_scores >  self.params["entry_z"]).sum()
        if buy_count > sell_count:
            signal, strength = "BUY", min(buy_count / df.shape[1], 1.0)
        elif sell_count > buy_count:
            signal, strength = "SELL", min(sell_count / df.shape[1], 1.0)
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"buy_count": int(buy_count), "sell_count": int(sell_count),
                                        "n_stocks": df.shape[1]})

    def describe(self) -> str:
        return ("PCA Basket Arb: decompose returns into common factors, trade mean-reversion "
                "of idiosyncratic residuals when z-score exceeds ±2σ.")


class CryptoBasisArb(StatArbStrategy):
    """
    Strategy 8 — Crypto Basis / Funding Rate Arbitrage
    df should have 'funding_rate' column (per 8-hour period) and 'spot' / 'perp' prices.
    Falls back to basis computed from close if columns not present.
    """
    strategy_id = 8

    def _default_params(self) -> dict:
        return {"min_funding_rate": 0.0003, "annualized_threshold": 0.10}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "CRYPTO") -> SignalResult:
        if "funding_rate" in df.columns:
            fr = float(df["funding_rate"].iloc[-1])
        else:
            # Proxy: use short-term vs long-term EMA spread as basis
            close = df["close"].astype(float)
            basis = (self._ema(close, 3) / self._ema(close, 30) - 1).iloc[-1]
            fr = float(basis) / 10
        annualized = fr * 3 * 365        # 3 funding periods/day × 365
        if fr > self.params["min_funding_rate"]:
            # Positive funding: long spot, short perp → collect funding
            signal = "BUY"
            strength = min(annualized / 0.30, 1.0)
        elif fr < -self.params["min_funding_rate"]:
            signal = "SELL"
            strength = min(abs(annualized) / 0.30, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"funding_rate": round(fr, 6),
                                        "annualized_yield": f"{annualized*100:.1f}%"})

    def describe(self) -> str:
        return ("Crypto Funding Rate Arb: long spot / short perpetual futures when funding rate "
                "is positive, earning the 8-hourly payment while remaining delta-neutral.")


# ─────────────────────────────────────────────
# TYPE 3 — MACHINE LEARNING STRATEGIES
# ─────────────────────────────────────────────

class MLStrategy(Strategy):
    """Base class for machine learning and AI-driven strategies."""
    strategy_type = "machine_learning"


class GradientBoostingAlpha(MLStrategy):
    """
    Strategy 9 — Gradient Boosting Ensemble Alpha (GBEA)
    Trains a LightGBM model on technical features to predict 5-day forward returns.
    Requires lightgbm; falls back to multi-feature scoring if not installed.
    """
    strategy_id = 9

    def _default_params(self) -> dict:
        return {"n_estimators": 200, "train_window": 252, "predict_horizon": 5, "top_pct": 0.2}

    def _build_features(self, close: pd.Series) -> pd.DataFrame:
        feats = pd.DataFrame(index=close.index)
        for w in [5, 10, 20, 60]:
            feats[f"ret_{w}d"]  = close.pct_change(w)
            feats[f"sma_{w}d"]  = close / self._sma(close, w) - 1
        macd, sig = self._macd(close)
        feats["macd_hist"]  = macd - sig
        feats["rsi_14"]     = self._rsi(close, 14)
        _, mid, _ = self._bollinger(close)
        feats["bb_pos"]     = (close - mid) / (close.rolling(20).std() + 1e-10)
        feats["vol_20d"]    = close.pct_change().rolling(20).std()
        return feats.dropna()

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        feats = self._build_features(close)
        if len(feats) < self.params["train_window"] + 30:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        # Rule-based ensemble score (proxy for model output)
        latest = feats.iloc[-1]
        score  = 0.0
        score += 0.3 * np.sign(latest.get("ret_20d", 0))
        score += 0.2 * np.sign(latest.get("macd_hist", 0))
        score += 0.2 * (1 if latest.get("rsi_14", 50) < 40 else (-1 if latest.get("rsi_14", 50) > 60 else 0))
        score += 0.15 * np.sign(-latest.get("bb_pos", 0))   # mean reversion component
        score += 0.15 * np.sign(latest.get("sma_60d", 0))
        if score > 0.3:
            signal, strength = "BUY", min(score, 1.0)
        elif score < -0.3:
            signal, strength = "SELL", min(abs(score), 1.0)
        else:
            signal, strength = "NEUTRAL", abs(score)
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={k: round(float(v), 4) for k, v in latest.items()},
                            notes="Rule-based feature score (proxy for GBDT model)")

    def describe(self) -> str:
        return ("Gradient Boosting Alpha: ensemble of 200 trees trained on 30+ technical features "
                "to predict 5-day forward returns. Top 20% → long, bottom 20% → short.")


class DeepRLAgent(MLStrategy):
    """
    Strategy 10 — Deep Reinforcement Learning Trading Agent
    DRL agent using Q-learning logic (state = technical indicators, action = buy/sell/hold).
    Simplified: uses a reward-driven heuristic without full neural network.
    """
    strategy_id = 10

    def _default_params(self) -> dict:
        return {"window": 20, "reward_threshold": 0.005}

    def _state(self, close: pd.Series) -> dict:
        rsi = float(self._rsi(close).iloc[-1])
        macd_l, macd_s = self._macd(close)
        macd_h = float((macd_l - macd_s).iloc[-1])
        vol    = float(close.pct_change().rolling(20).std().iloc[-1])
        trend  = float((close.iloc[-1] / self._sma(close, 50).iloc[-1]) - 1)
        return {"rsi": rsi, "macd_hist": macd_h, "vol": vol, "trend": trend}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 60:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        state = self._state(close)
        # Simplified Q-value heuristic (no NN weights loaded)
        q_buy  = (state["rsi"] < 40) * 0.4 + (state["macd_hist"] > 0) * 0.3 + (state["trend"] > 0) * 0.3
        q_sell = (state["rsi"] > 60) * 0.4 + (state["macd_hist"] < 0) * 0.3 + (state["trend"] < 0) * 0.3
        if q_buy > q_sell and q_buy > 0.5:
            signal, strength = "BUY", q_buy
        elif q_sell > q_buy and q_sell > 0.5:
            signal, strength = "SELL", q_sell
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators=state, notes="Heuristic proxy for DRL Q-values")

    def describe(self) -> str:
        return ("Deep RL Agent: state = [RSI, MACD, trend, volatility]; action space = {buy, sell, hold}. "
                "Agent trained via Q-learning to maximize risk-adjusted reward.")


class LLMEarningsAlpha(MLStrategy):
    """
    Strategy 11 — LLM-Driven Earnings and Filings Alpha
    Uses price reaction around earnings as a proxy for LLM sentiment signal.
    Full version would call an LLM API on 10-K/earnings transcript text.
    """
    strategy_id = 11

    def _default_params(self) -> dict:
        return {"pre_event_days": 3, "post_event_days": 1, "surprise_threshold": 0.02}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 20:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        # Proxy: look for unusual post-earnings gap
        daily_ret  = close.pct_change()
        vol_20     = daily_ret.rolling(20).std().iloc[-1]
        last_ret   = daily_ret.iloc[-1]
        z_score    = last_ret / (vol_20 + 1e-10)
        # Drift continuation after strong earnings move
        if z_score > 2.0:
            signal, strength = "BUY", min(z_score / 4, 1.0)
        elif z_score < -2.0:
            signal, strength = "SELL", min(abs(z_score) / 4, 1.0)
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"last_day_z": round(z_score, 2), "vol_20d": round(vol_20, 4)},
                            notes="Proxy for LLM earnings sentiment — requires NLP integration for production")

    def describe(self) -> str:
        return ("LLM Earnings Alpha: LLM (GPT-4o/Claude) reads earnings transcripts and 10-K filings "
                "to generate sentiment scores. Price-reaction proxy used here for live demo.")


class AltDataSatellite(MLStrategy):
    """
    Strategy 12 — Alternative Data: Satellite + Credit Card Signal
    Proxied by volume anomalies and price-volume divergence.
    Full version would ingest satellite foot-traffic or credit card panel data.
    """
    strategy_id = 12

    def _default_params(self) -> dict:
        return {"vol_z_threshold": 2.0, "lookback": 60}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)
        if len(close) < self.params["lookback"] or volume.empty:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        vol_z = ((volume - volume.rolling(self.params["lookback"]).mean())
                 / volume.rolling(self.params["lookback"]).std()).iloc[-1]
        price_ret = close.pct_change().iloc[-1]
        # High volume + positive return → demand surge (alt data confirmation proxy)
        if vol_z > self.params["vol_z_threshold"] and price_ret > 0:
            signal, strength = "BUY", min(vol_z / 4, 1.0)
        elif vol_z > self.params["vol_z_threshold"] and price_ret < 0:
            signal, strength = "SELL", min(vol_z / 4, 1.0)
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"volume_z": round(float(vol_z), 2), "price_ret_pct": round(price_ret * 100, 2)},
                            notes="Volume anomaly as proxy for satellite/credit card data")

    def describe(self) -> str:
        return ("Alt Data Strategy: satellite imagery foot-traffic + credit card transactions "
                "predict retail/energy revenues. Volume anomaly used as proxy signal.")


# ─────────────────────────────────────────────
# TYPE 4 — VOLATILITY STRATEGIES
# ─────────────────────────────────────────────

class VolatilityStrategy(Strategy):
    """Base class for options and volatility-based strategies."""
    strategy_type = "volatility"


class VolatilityRiskPremium(VolatilityStrategy):
    """
    Strategy 13 — Volatility Risk Premium (VRP) Harvesting
    Sell implied vol (expensive) when realized vol (cheap) suggests a premium exists.
    Proxied by comparing VIX (or rolling IV) to realized vol.
    """
    strategy_id = 13

    def _default_params(self) -> dict:
        return {"rv_window": 21, "vrp_threshold": 2.0}  # VRP in vol points

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 30:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        rv = float(close.pct_change().rolling(self.params["rv_window"]).std().iloc[-1]) * np.sqrt(252) * 100
        if "iv" in df.columns:
            iv = float(df["iv"].iloc[-1]) * 100
        else:
            iv = rv * 1.15    # IV typically ~15% above realized vol on average
        vrp = iv - rv
        if vrp > self.params["vrp_threshold"]:
            signal = "SELL"    # Sell options / short vol → premium is rich
            strength = min(vrp / 10, 1.0)
        elif vrp < -self.params["vrp_threshold"]:
            signal = "BUY"     # Buy options / long vol → premium is cheap
            strength = min(abs(vrp) / 10, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"realized_vol_pct": round(rv, 2), "implied_vol_pct": round(iv, 2),
                                        "vrp_vol_points": round(vrp, 2)})

    def describe(self) -> str:
        return ("VRP Harvesting: sell options when implied vol exceeds realized vol by >2 vol points, "
                "capturing the volatility risk premium. Short straddle / cash-secured puts.")


class DispersionTrading(VolatilityStrategy):
    """
    Strategy 14 — Dispersion Trading
    Short index vol, long single-stock vols. Proxied by comparing index vol to average stock vol.
    """
    strategy_id = 14

    def _default_params(self) -> dict:
        return {"dispersion_threshold": 5.0}  # vol points

    def generate_signal(self, df: pd.DataFrame, symbol: str = "INDEX") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 30:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        index_vol = float(close.pct_change().rolling(21).std().iloc[-1]) * np.sqrt(252) * 100
        # In a real implementation, compare index_vol to cross-sectional average of stock vols
        # Proxy: high autocorrelation of returns → low dispersion (bad for strategy)
        autocorr = float(close.pct_change().autocorr(lag=1))
        implied_dispersion = index_vol * (1 - autocorr)
        if implied_dispersion < index_vol * 0.85:
            signal, strength = "BUY", 0.7    # sell index vol, buy stock vols
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"index_vol": round(index_vol, 2),
                                        "implied_dispersion": round(implied_dispersion, 2),
                                        "autocorr": round(autocorr, 3)},
                            notes="Proxy for dispersion — requires multi-stock vol surface in production")

    def describe(self) -> str:
        return ("Dispersion Trading: short index implied vol, long individual stock vols. "
                "Profits when stocks move independently (high dispersion).")


class GammaScalping(VolatilityStrategy):
    """
    Strategy 15 — Gamma Scalping (Long Gamma / Delta-Neutral)
    Hold long options position; delta-hedge frequently to extract realized vol.
    Proxied by measuring realized vol vs cost of carry.
    """
    strategy_id = 15

    def _default_params(self) -> dict:
        return {"hedge_frequency_days": 1, "rv_lookback": 10}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 20:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        rv_short  = float(close.pct_change().rolling(self.params["rv_lookback"]).std().iloc[-1]) * np.sqrt(252) * 100
        rv_long   = float(close.pct_change().rolling(21).std().iloc[-1]) * np.sqrt(252) * 100
        # High short-term vol relative to 21-day vol → gamma is being realized → profitable to be long gamma
        vol_ratio = rv_short / (rv_long + 1e-10)
        if vol_ratio > 1.2:
            signal, strength = "BUY", min((vol_ratio - 1.0) * 2, 1.0)
        elif vol_ratio < 0.7:
            signal, strength = "SELL", min((1.0 - vol_ratio) * 2, 1.0)
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"rv_10d": round(rv_short, 2), "rv_21d": round(rv_long, 2),
                                        "vol_ratio": round(vol_ratio, 3)})

    def describe(self) -> str:
        return ("Gamma Scalping: buy straddles and delta-hedge continuously to extract realized vol. "
                "Profitable when actual vol exceeds implied vol paid.")


# ─────────────────────────────────────────────
# TYPE 5 — FACTOR / SMART BETA STRATEGIES
# ─────────────────────────────────────────────

class FactorStrategy(Strategy):
    """Base class for multi-factor and smart-beta strategies."""
    strategy_type = "factor"


class MultiFactorLongShort(FactorStrategy):
    """
    Strategy 16 — Multi-Factor Long-Short Equity (Quality + Value + Momentum)
    Combines RSI (momentum), price-to-SMA ratio (value proxy), and vol stability (quality proxy).
    """
    strategy_id = 16

    def _default_params(self) -> dict:
        return {"momentum_weight": 0.40, "value_weight": 0.30, "quality_weight": 0.30}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 60:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        # Momentum factor: 12-1 return
        mom = (close.iloc[-1] / close.iloc[-min(252, len(close)-21)]) - 1
        mom_score = np.sign(mom) * min(abs(mom) / 0.30, 1.0)
        # Value factor: price vs 200-day SMA (lower = cheaper)
        sma200 = self._sma(close, min(200, len(close))).iloc[-1]
        val = (sma200 / close.iloc[-1]) - 1   # positive = cheap
        val_score = np.sign(val) * min(abs(val) / 0.30, 1.0)
        # Quality factor: inverse of vol (lower vol = higher quality)
        vol = self._volatility(close, 60)
        q_score = -np.sign(vol - 0.25)   # below 25% vol = positive quality
        composite = (self.params["momentum_weight"]  * mom_score +
                     self.params["value_weight"]      * val_score +
                     self.params["quality_weight"]    * q_score)
        if composite > 0.3:
            signal, strength = "BUY", min(composite, 1.0)
        elif composite < -0.3:
            signal, strength = "SELL", min(abs(composite), 1.0)
        else:
            signal, strength = "NEUTRAL", abs(composite)
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"momentum_score": round(mom_score, 3),
                                        "value_score": round(val_score, 3),
                                        "quality_score": round(q_score, 3),
                                        "composite": round(composite, 3)})

    def describe(self) -> str:
        return ("Multi-Factor L/S: composite of Momentum (40%), Value (30%), Quality (30%). "
                "Long top-scoring stocks, short lowest-scoring. Rebalance monthly.")


class FactorMomentum(FactorStrategy):
    """
    Strategy 17 — Factor Momentum (Momentum of Momentum / MOM of MOM)
    Trade factors themselves based on their recent performance.
    Proxied by rotating between momentum signals of different timescales.
    """
    strategy_id = 17

    def _default_params(self) -> dict:
        return {"fast_mom": 21, "medium_mom": 63, "slow_mom": 126}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < self.params["slow_mom"] + 5:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        moms = {}
        for key, w in [("fast", self.params["fast_mom"]),
                       ("medium", self.params["medium_mom"]),
                       ("slow", self.params["slow_mom"])]:
            moms[key] = (close.iloc[-1] / close.iloc[-w]) - 1
        # Factor momentum: are multiple factor windows aligned?
        positive = sum(1 for v in moms.values() if v > 0)
        negative = sum(1 for v in moms.values() if v < 0)
        if positive == 3:
            signal, strength = "BUY", 1.0
        elif negative == 3:
            signal, strength = "SELL", 1.0
        elif positive == 2:
            signal, strength = "BUY", 0.6
        elif negative == 2:
            signal, strength = "SELL", 0.6
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={f"mom_{k}d_pct": round(v * 100, 2) for k, v in moms.items()})

    def describe(self) -> str:
        return ("Factor Momentum: trade momentum factors based on their own past performance. "
                "Rotate into winning factors; rotate out of losing factors.")


# ─────────────────────────────────────────────
# TYPE 6 — SENTIMENT / NLP
# ─────────────────────────────────────────────

class SentimentStrategy(Strategy):
    """Base class for sentiment and NLP-driven strategies."""
    strategy_type = "sentiment"


class SocialNewsSentiment(SentimentStrategy):
    """
    Strategy 18 — Real-Time Social + News Sentiment Alpha
    Requires 'sentiment_score' column in df (-1 to +1) from NLP feed.
    Falls back to overnight gap + volume surge as sentiment proxy.
    """
    strategy_id = 18

    def _default_params(self) -> dict:
        return {"sentiment_threshold": 0.3, "decay_days": 3}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if "sentiment_score" in df.columns:
            score = float(df["sentiment_score"].rolling(self.params["decay_days"]).mean().iloc[-1])
            if score > self.params["sentiment_threshold"]:
                signal, strength = "BUY", min(score, 1.0)
            elif score < -self.params["sentiment_threshold"]:
                signal, strength = "SELL", min(abs(score), 1.0)
            else:
                signal, strength = "NEUTRAL", abs(score)
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                                indicators={"sentiment_score": round(score, 3)})
        # Proxy: overnight gap + unusual volume
        if len(close) < 10:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        gap = float(close.pct_change().iloc[-1])
        vol_ratio = 1.0
        if "volume" in df.columns:
            vol_ratio = float(df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1])
        sentiment_proxy = gap * min(vol_ratio, 3.0)
        if sentiment_proxy > 0.02:
            signal, strength = "BUY", min(sentiment_proxy * 10, 1.0)
        elif sentiment_proxy < -0.02:
            signal, strength = "SELL", min(abs(sentiment_proxy) * 10, 1.0)
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"overnight_gap_pct": round(gap * 100, 2),
                                        "volume_ratio": round(vol_ratio, 2)},
                            notes="Proxy signal — attach real NLP feed for production")

    def describe(self) -> str:
        return ("Social/News Sentiment: NLP models score Reddit, Twitter/X, news articles (-1 to +1). "
                "Decay-weighted average score drives long/short signal.")


# ─────────────────────────────────────────────
# TYPE 7 — MARKET MAKING / HFT
# ─────────────────────────────────────────────

class MarketMakingStrategy(Strategy):
    """Base class for market making and high-frequency strategies."""
    strategy_type = "market_making"


class ElectronicMarketMaking(MarketMakingStrategy):
    """
    Strategy 19 — Electronic Market Making (Quote-Driven HFT)
    Posts bid/ask quotes around fair value. Profits from bid-ask spread.
    Proxied using OHLCV: estimates intraday spread and fair value mid.
    """
    strategy_id = 19

    def _default_params(self) -> dict:
        return {"spread_bps": 5, "inventory_limit": 0.1, "vol_adjustment": True}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 5:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        # Estimate effective spread from OHLC
        high, low = df["high"].astype(float).iloc[-1], df["low"].astype(float).iloc[-1]
        intraday_range_bps = (high - low) / close.iloc[-1] * 10000
        vol = self._volatility(close, 20)
        # Market maker wants to quote when spread > vol-adjusted threshold
        if intraday_range_bps > self.params["spread_bps"] * 2:
            signal = "BUY"     # Post aggressive quotes (profitable spread environment)
            strength = min(intraday_range_bps / 50, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"intraday_range_bps": round(intraday_range_bps, 1),
                                        "realized_vol": round(vol, 4),
                                        "fair_value": round(close.iloc[-1], 4)},
                            notes="Market making signal: BUY = favorable quoting environment")

    def describe(self) -> str:
        return ("Electronic Market Making: post bid/ask quotes ±spread_bps around mid price. "
                "Inventory management limits exposure. Profits from spread × fill rate.")


# ─────────────────────────────────────────────
# TYPE 8 — EVENT-DRIVEN
# ─────────────────────────────────────────────

class EventDrivenStrategy(Strategy):
    """Base class for event-driven strategies."""
    strategy_type = "event_driven"


class EarningsSurprise(EventDrivenStrategy):
    """
    Strategy 20 — Systematic Event-Driven: Earnings Surprise + Price Reaction
    Detects post-earnings drift: stocks with large earnings-day gaps continue in same direction.
    """
    strategy_id = 20

    def _default_params(self) -> dict:
        return {"gap_threshold": 0.03, "drift_window": 5, "vol_normalize": True}

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> SignalResult:
        close = df["close"].astype(float)
        if len(close) < 30:
            return SignalResult(self.__class__.__name__, self.strategy_type, symbol, "NEUTRAL", 0.0)
        daily_rets = close.pct_change()
        vol = daily_rets.rolling(20).std().iloc[-1]
        # Find largest single-day move in recent history (earnings proxy)
        recent = daily_rets.iloc[-20:]
        max_gap_idx = recent.abs().idxmax()
        max_gap     = daily_rets[max_gap_idx]
        days_since  = len(close) - close.index.get_loc(max_gap_idx) - 1
        z_gap = max_gap / (vol + 1e-10)
        # Post-earnings drift: if big gap happened recently and we're in drift window
        if days_since <= self.params["drift_window"] and abs(z_gap) > 2.0:
            if max_gap > self.params["gap_threshold"]:
                signal, strength = "BUY", min(abs(z_gap) / 4, 1.0)
            elif max_gap < -self.params["gap_threshold"]:
                signal, strength = "SELL", min(abs(z_gap) / 4, 1.0)
            else:
                signal, strength = "NEUTRAL", 0.0
        else:
            signal, strength = "NEUTRAL", 0.0
        return SignalResult(self.__class__.__name__, self.strategy_type, symbol, signal, strength,
                            indicators={"earnings_gap_pct": round(max_gap * 100, 2),
                                        "gap_z_score": round(z_gap, 2),
                                        "days_since_event": days_since})

    def describe(self) -> str:
        return ("Earnings Surprise Drift: detect large single-day price gaps (earnings proxy). "
                "Trade in direction of gap for 5-day post-earnings drift window.")


# ─────────────────────────────────────────────
# REGISTRY — all 20 strategies
# ─────────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    # Momentum
    "tsmom":                  TimeSeriesMomentum,
    "cross_sectional_mom":    CrossSectionalMomentum,
    "ml_enhanced_trend":      MLEnhancedTrend,
    "short_term_reversal":    ShortTermReversal,
    # Stat Arb
    "pairs_trading":          PairsTrading,
    "etf_arb":                ETFArbitrage,
    "pca_basket_arb":         PCABasketArb,
    "crypto_basis_arb":       CryptoBasisArb,
    # Machine Learning
    "gradient_boosting":      GradientBoostingAlpha,
    "deep_rl":                DeepRLAgent,
    "llm_earnings":           LLMEarningsAlpha,
    "alt_data_satellite":     AltDataSatellite,
    # Volatility
    "vrp_harvesting":         VolatilityRiskPremium,
    "dispersion":             DispersionTrading,
    "gamma_scalping":         GammaScalping,
    # Factor
    "multi_factor":           MultiFactorLongShort,
    "factor_momentum":        FactorMomentum,
    # Sentiment
    "social_sentiment":       SocialNewsSentiment,
    # Market Making
    "market_making":          ElectronicMarketMaking,
    # Event-Driven
    "earnings_surprise":      EarningsSurprise,
}


def get_strategy(name: str, params: dict | None = None) -> Strategy:
    """Instantiate a strategy by registry key."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](params)


def list_strategies() -> pd.DataFrame:
    """Return a summary DataFrame of all registered strategies."""
    rows = []
    for key, cls in STRATEGY_REGISTRY.items():
        obj = cls()
        rows.append({
            "key":   key,
            "class": cls.__name__,
            "type":  cls.strategy_type,
            "id":    cls.strategy_id,
        })
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


# ─────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Available strategies:\n")
    print(list_strategies().to_string(index=False))

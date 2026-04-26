import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore
from scipy.optimize import brentq, minimize_scalar
from dataclasses import dataclass, field
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

@dataclass
class AssetConfig:
    name:           str
    asset_class:    Literal["fx", "equity_index"]
    ticker:         str           
    pip_size:       float         # 0.0001 for FX majors, 1.0 for indexes
    round_grid:     float         # strike grid for round numbers
    typical_iv:     float         # annualized IV, decimal (0.08 = 8%)
    vol_of_vol:     float         # how much IV itself moves day-to-day
    skew_adj:       float         # put wing premium vs ATM (0 = symmetric)
    min_notional:   float         # smallest liquid options contract (USD equiv)
    bars_per_year:  int           # 252*24 hourly, 252 daily


ASSETS = {
    "EURUSD": AssetConfig(
        name="EUR/USD", asset_class="fx", ticker="EURUSD=X",
        pip_size=0.0001, round_grid=0.0050,
        typical_iv=0.07, vol_of_vol=0.18, skew_adj=0.00,
        min_notional=1_000_000, bars_per_year=252 * 24,
    ),
    "USDJPY": AssetConfig(
        name="USD/JPY", asset_class="fx", ticker="USDJPY=X",
        pip_size=0.01, round_grid=0.50,
        typical_iv=0.08, vol_of_vol=0.22, skew_adj=0.01,
        min_notional=1_000_000, bars_per_year=252 * 24,
    ),
    "GBPUSD": AssetConfig(
        name="GBP/USD", asset_class="fx", ticker="GBPUSD=X",
        pip_size=0.0001, round_grid=0.0050,
        typical_iv=0.08, vol_of_vol=0.20, skew_adj=0.00,
        min_notional=1_000_000, bars_per_year=252 * 24,
    ),
    "SPX": AssetConfig(
        name="S&P 500", asset_class="equity_index", ticker="^GSPC",
        pip_size=1.0, round_grid=50.0,
        typical_iv=0.17, vol_of_vol=0.55, skew_adj=0.04,
        min_notional=100_000, bars_per_year=252 * 24,
    ),
    "IBOV": AssetConfig(
        name="Bovespa", asset_class="equity_index", ticker="^BVSP",
        pip_size=1.0, round_grid=1000.0,
        typical_iv=0.22, vol_of_vol=0.45, skew_adj=0.03,
        min_notional=50_000, bars_per_year=252 * 24,
    ),
}

#  BLACK-SCHOLES GREEKS

def bs_d1d2(S, K, T, sigma, r=0.0):
    if T <= 1e-8 or sigma <= 1e-8:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S, K, T, sigma, opt="call", r=0.0):
    d1, d2 = bs_d1d2(S, K, T, sigma, r)
    if np.isnan(d1):
        return max(S - K, 0) if opt == "call" else max(K - S, 0)
    if opt == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, sigma):
    """Gamma: ∂²V/∂S². Units: per dollar² of spot move."""
    d1, _ = bs_d1d2(S, K, T, sigma)
    if np.isnan(d1):
        return 0.0
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S, K, T, sigma, r=0.0):
    """Theta: daily time decay (annualized / 365)."""
    d1, d2 = bs_d1d2(S, K, T, sigma, r)
    if np.isnan(d1):
        return 0.0
    return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) / 365


def bs_vanna(S, K, T, sigma):
    """Vanna: ∂Gamma/∂sigma. Measures gamma sensitivity to vol changes."""
    d1, d2 = bs_d1d2(S, K, T, sigma)
    if np.isnan(d1):
        return 0.0
    return -norm.pdf(d1) * d2 / sigma


def straddle_greeks(S, K, T, sigma):
    g = bs_gamma(S, K, T, sigma)
    th = 2 * bs_theta(S, K, T, sigma)
    prem = bs_price(S, K, T, sigma, "call") + bs_price(S, K, T, sigma, "put")
    van = bs_vanna(S, K, T, sigma)
    return {"gamma": g, "theta": th, "premium": prem, "vanna": van}

#  1: GAMMA-THETA BREAKEVEN ANALYSIS

def gamma_pnl_from_squeeze(
    S: float,
    K: float,
    T: float,
    sigma_iv: float,
    squeeze_pips: float,
    notional: float,
    cfg: AssetConfig,
) -> dict:
    """
    Estimate gamma P&L from a squeeze of squeeze_pips toward the strike.

    Uses: P&L ≈ ½ × Γ × N × (ΔS)²
    This is the instantaneous gamma scalp — assumes you don't re-delta-hedge
    during the squeeze (you let the gamma run). For a hedged position,
    multiply by the number of hedging intervals.

    Returns detailed breakdown including theta cost estimate.
    """
    g = bs_gamma(S, K, T, sigma_iv)
    greeks = straddle_greeks(S, K, T, sigma_iv)

    delta_s = squeeze_pips * cfg.pip_size
    gamma_pnl = 0.5 * g * notional * delta_s**2

    # Theta cost for holding T_hold days
    theta_daily = abs(greeks["theta"]) * notional
    prem_total = greeks["premium"] * notional

    # Time for squeeze to materialize (rough estimate: 1-3 days)
    squeeze_days_est = delta_s / (S * sigma_iv / np.sqrt(252))
    theta_cost_est = theta_daily * max(squeeze_days_est, 0.5)

    net_pnl_est = gamma_pnl - theta_cost_est

    return {
        "spot": S,
        "strike": K,
        "dist_pips": round((S - K) / cfg.pip_size, 1),
        "bs_gamma": round(g, 8),
        "dollar_gamma": round(g * notional * S**2 * 0.01, 2),  # P&L per 1% spot move
        "gamma_pnl_gross": round(gamma_pnl, 2),
        "theta_daily_usd": round(theta_daily, 2),
        "squeeze_days_est": round(squeeze_days_est, 2),
        "theta_cost_est": round(theta_cost_est, 2),
        "net_pnl_est": round(net_pnl_est, 2),
        "prem_total": round(prem_total, 2),
        "breakeven_squeeze_pips": round(
            np.sqrt(2 * theta_daily / (g * notional)) / cfg.pip_size, 1
        ) if g > 0 else np.inf,
    }


def gamma_theta_breakeven_surface(
    cfg: AssetConfig,
    S: float,
    T: float = 5 / 365,
    notional: float = 1_000_000,
    dist_range: tuple = (10, 150),
    iv_range: tuple = (0.05, 0.20),
) -> pd.DataFrame:
    """
    For each (distance from strike, IV level) pair, compute:
      - gamma load
      - theta cost per day
      - minimum RV needed for gamma scalping to be profitable
      - breakeven squeeze in pips

    This surface tells you WHERE on the IV/distance grid long gamma has edge.
    """
    dists = np.arange(dist_range[0], dist_range[1] + 1, 10)
    ivs   = np.arange(iv_range[0], iv_range[1] + 0.01, 0.01)
    rows  = []

    for dist_pips in dists:
        K = S - dist_pips * cfg.pip_size  # spot above strike
        for iv in ivs:
            g      = bs_gamma(S, K, T, iv)
            th     = abs(bs_theta(S, K, T, iv)) * notional
            prem   = (bs_price(S, K, T, iv, "call") +
                      bs_price(S, K, T, iv, "put")) * notional

            # Min RV for breakeven: ½ × Γ × N × (rv×S×√(1/252))² = theta/252
            # → rv = sqrt(2 * theta_annual / (Γ × N × S²))
            theta_annual = th * 365
            if g > 0 and notional > 0:
                min_rv = np.sqrt(2 * theta_annual / (g * notional * S**2))
            else:
                min_rv = np.inf

            rows.append({
                "dist_pips": dist_pips,
                "iv": round(iv * 100, 1),
                "gamma": round(g, 8),
                "dollar_gamma_per_pct": round(g * notional * S**2 * 0.01, 0),
                "theta_usd_day": round(th, 0),
                "prem_usd": round(prem, 0),
                "min_rv_pct": round(min_rv * 100, 2),
                "iv_minus_min_rv": round((iv - min_rv) * 100, 2),
                "edge_exists": iv > min_rv,  # True = theta drag < gamma earn at current RV
            })

    return pd.DataFrame(rows)


#  2: OPTIMAL GAMMA SIZING

def optimal_notional_for_theta_budget(
    S: float,
    K: float,
    T: float,
    sigma_iv: float,
    squeeze_pips: float,
    theta_budget_usd: float,
    cfg: AssetConfig,
) -> dict:
    """
    Find the notional such that:
      daily theta cost = theta_budget_usd
    Then compute expected gamma P&L at that notional for the expected squeeze.

    This is your position sizing anchor — you start from what you can afford
    to lose in theta per day, then check if the gamma P&L is worth it.
    """
    th_per_unit = abs(bs_theta(S, K, T, sigma_iv))  # per unit notional
    if th_per_unit <= 0:
        return {"error": "theta is zero"}

    # Notional where theta = budget
    notional = theta_budget_usd / th_per_unit

    # Clamp to minimum contract size
    notional = max(notional, cfg.min_notional)

    result = gamma_pnl_from_squeeze(
        S, K, T, sigma_iv, squeeze_pips, notional, cfg
    )
    result["theta_budget_usd"] = theta_budget_usd
    result["optimal_notional"] = round(notional, 0)
    result["pnl_to_theta_ratio"] = round(
        result["gamma_pnl_gross"] / theta_budget_usd, 2
    )
    return result


def gamma_sizing_by_regime(
    S: float,
    K: float,
    T: float,
    sigma_iv: float,
    rv_series: pd.Series,
    cfg: AssetConfig,
    theta_budget_usd: float = 1000,
    squeeze_pips: float = 50,
) -> pd.DataFrame:
    """
    Compute optimal gamma sizing across different vol regimes.
    Regimes defined by RV percentile over the lookback.

    Key insight: when RV >> IV (vol risk premium is negative = vol cheap),
    you want MORE gamma. When IV >> RV (vol expensive), reduce gamma load.

    Returns a DataFrame with sizing recommendations per regime.
    """
    rv_annual = rv_series.dropna()
    pct_now   = percentileofscore(rv_annual, rv_annual.iloc[-1])
    iv_pct    = percentileofscore(rv_annual, sigma_iv)

    regimes = {
        "low_vol":     rv_annual.quantile(0.25),
        "median_vol":  rv_annual.quantile(0.50),
        "high_vol":    rv_annual.quantile(0.75),
        "stress_vol":  rv_annual.quantile(0.90),
    }

    rows = []
    for regime_name, rv_est in regimes.items():
        rv_iv_spread = rv_est - sigma_iv

        # Gamma multiplier: scale up when RV > IV (positive carry on gamma)
        # Scale down when IV > RV (paying too much for gamma relative to realized)
        if rv_iv_spread > 0:
            gamma_mult = 1.0 + min(rv_iv_spread / sigma_iv, 1.5)  # cap at 2.5x
        else:
            gamma_mult = max(1.0 + rv_iv_spread / sigma_iv, 0.25)  # floor at 0.25x

        # Asset-class adjustments
        if cfg.asset_class == "equity_index":
            # Equity index: in high-vol regimes gamma is much more expensive
            # (vol-of-vol is high, so gamma itself is volatile — harder to hold)
            if rv_est > cfg.typical_iv * 1.5:
                gamma_mult *= 0.7  # reduce in stress: gaps make hedging hard
            # Skew adjustment: in equity, prefer buying OTM calls over puts near round
            # numbers because put-side gamma is systematically overpriced (skew premium)
            gamma_mult *= (1 - cfg.skew_adj)

        adj_budget = theta_budget_usd * gamma_mult
        sizing = optimal_notional_for_theta_budget(
            S, K, T, sigma_iv, squeeze_pips, adj_budget, cfg
        )

        rows.append({
            "regime":           regime_name,
            "rv_est_pct":       round(rv_est * 100, 2),
            "iv_pct":           round(sigma_iv * 100, 2),
            "rv_iv_spread_pct": round(rv_iv_spread * 100, 2),
            "gamma_multiplier": round(gamma_mult, 2),
            "adj_theta_budget": round(adj_budget, 0),
            "optimal_notional": sizing.get("optimal_notional", 0),
            "dollar_gamma":     sizing.get("dollar_gamma_per_pct", 0),
            "gamma_pnl_gross":  sizing.get("gamma_pnl_gross", 0),
            "net_pnl_est":      sizing.get("net_pnl_est", 0),
            "pnl_theta_ratio":  sizing.get("pnl_to_theta_ratio", 0),
        })

    return pd.DataFrame(rows)


#  MODULE 3: GAMMA STRUCTURE COMPARISON
#  1. ATM straddle
#  2. Strangle (±N pips)
#  3. 1x2 call spread
#  4. Gamma scalp via dynamic hedging

def compare_gamma_structures(
    S: float,
    K: float,
    T: float,
    sigma_iv: float,
    notional: float,
    cfg: AssetConfig,
    strangle_width_pips: float = 50,
) -> pd.DataFrame:
    """
    Compare gamma, theta, pnl profiles for different option structures.
    All sized to the same notional for apples-to-apples comparison.
    """
    pip = cfg.pip_size
    K_call = K + strangle_width_pips * pip  # OTM call for strangle
    K_put  = K - strangle_width_pips * pip  # OTM put for strangle
    K2     = K + strangle_width_pips * pip  # short call for 1x2

    structures = {}

    # 1. ATM straddle
    g_atm = bs_gamma(S, K, T, sigma_iv)
    th_atm = 2 * abs(bs_theta(S, K, T, sigma_iv))
    prem_atm = (bs_price(S, K, T, sigma_iv, "call") +
                bs_price(S, K, T, sigma_iv, "put"))
    structures["ATM straddle"] = {
        "gamma_unit": g_atm,
        "theta_unit": th_atm,
        "prem_unit":  prem_atm,
        "net_gamma_sign": "+",
        "comment": "Max gamma at strike; highest theta; ideal for pin squeeze",
    }

    # 2. Strangle
    g_str = bs_gamma(S, K_call, T, sigma_iv) + bs_gamma(S, K_put, T, sigma_iv)
    th_str = abs(bs_theta(S, K_call, T, sigma_iv)) + abs(bs_theta(S, K_put, T, sigma_iv))
    prem_str = (bs_price(S, K_call, T, sigma_iv, "call") +
                bs_price(S, K_put, T, sigma_iv, "put"))
    structures[f"Strangle ±{strangle_width_pips}p"] = {
        "gamma_unit": g_str,
        "theta_unit": th_str,
        "prem_unit":  prem_str,
        "net_gamma_sign": "+",
        "comment": "Lower cost; gamma profile flatter; good if squeeze uncertain",
    }

    # 3. 1x2 call spread (long 1 ATM call, short 2 OTM calls)
    g_long  = bs_gamma(S, K, T, sigma_iv)
    g_short = bs_gamma(S, K2, T, sigma_iv)
    g_1x2   = g_long - 2 * g_short
    th_1x2  = abs(bs_theta(S, K, T, sigma_iv)) - 2 * abs(bs_theta(S, K2, T, sigma_iv))
    prem_1x2 = (bs_price(S, K, T, sigma_iv, "call") -
                2 * bs_price(S, K2, T, sigma_iv, "call"))
    structures[f"1x2 call spread (ATM vs +{strangle_width_pips}p)"] = {
        "gamma_unit": g_1x2,
        "theta_unit": th_1x2,
        "prem_unit":  prem_1x2,
        "net_gamma_sign": "+ near K, − above K2",
        "comment": "Long gamma near pin; self-funding if OTM premium covers; cap above K2",
    }

    # 4. Call ratio (long 1 ATM, short 1.5 OTM) — intermediate
    g_ratio   = g_long - 1.5 * g_short
    th_ratio  = abs(bs_theta(S, K, T, sigma_iv)) - 1.5 * abs(bs_theta(S, K2, T, sigma_iv))
    prem_ratio = (bs_price(S, K, T, sigma_iv, "call") -
                  1.5 * bs_price(S, K2, T, sigma_iv, "call"))
    structures[f"1x1.5 ratio (ATM vs +{strangle_width_pips}p)"] = {
        "gamma_unit": g_ratio,
        "theta_unit": th_ratio,
        "prem_unit":  prem_ratio,
        "net_gamma_sign": "+ near K, − above K2",
        "comment": "Partial hedge of theta; leaves tail gamma; for moderate conviction",
    }

    rows = []
    for name, s in structures.items():
        dollar_gamma = s["gamma_unit"] * notional * S**2 * 0.01
        theta_usd    = s["theta_unit"] * notional
        prem_usd     = s["prem_unit"] * notional
        g_th_ratio   = dollar_gamma / abs(theta_usd) if theta_usd != 0 else np.inf

        rows.append({
            "structure":           name,
            "dollar_gamma_per_1pct": round(dollar_gamma, 0),
            "theta_usd_day":       round(theta_usd, 0),
            "prem_usd":            round(prem_usd, 0),
            "gamma_theta_ratio":   round(g_th_ratio, 3),
            "net_gamma_sign":      s["net_gamma_sign"],
            "comment":             s["comment"],
        })

    return pd.DataFrame(rows)

#  MODULE 4: CROSS-ASSET GAMMA SIZING TABLE

def cross_asset_gamma_table(
    theta_budget_usd: float,
    squeeze_map: dict,       # {asset_key: squeeze_pips}
    spot_map: dict,          # {asset_key: current_spot}
    iv_map: dict,            # {asset_key: current_iv (decimal)}
    dte: float = 5,
) -> pd.DataFrame:
    """
    Build a cross-asset gamma sizing table for a given theta budget.

    squeeze_map: expected squeeze in pips per asset (your judgment call)
    spot_map:    current spot for each asset
    iv_map:      current observed IV for each asset

    Returns recommended notional, dollar gamma, and expected P&L per asset.
    """
    T = dte / 365
    rows = []

    for key, cfg in ASSETS.items():
        if key not in spot_map:
            continue

        S  = spot_map[key]
        iv = iv_map.get(key, cfg.typical_iv)
        sq = squeeze_map.get(key, 50)

        # Round number strike (nearest grid below spot)
        K = round(round(S / cfg.round_grid) * cfg.round_grid, 6)
        dist_pips = (S - K) / cfg.pip_size

        # Liquidity haircut: reduce effective gamma budget for less liquid markets
        liq_haircut = {
            "EURUSD": 1.00,
            "USDJPY": 0.95,
            "GBPUSD": 0.90,
            "SPX":    0.85,   # index options liquid but wide skew
            "IBOV":   0.60,   # EM options: wider spreads, gap risk
        }.get(key, 0.80)

        adj_budget = theta_budget_usd * liq_haircut

        sizing = optimal_notional_for_theta_budget(
            S, K, T, iv, sq, adj_budget, cfg
        )

        # Vol risk premium signal: if IV > typical RV → gamma expensive → reduce
        vrp_signal = (iv - cfg.typical_iv) / cfg.typical_iv  # +ve = IV rich
        vrp_adj = max(1.0 - vrp_signal, 0.5)

        adj_notional = sizing.get("optimal_notional", 0) * vrp_adj

        rows.append({
            "asset":               cfg.name,
            "class":               cfg.asset_class,
            "spot":                S,
            "strike":              K,
            "dist_pips":           round(dist_pips, 1),
            "iv_pct":              round(iv * 100, 2),
            "typical_iv_pct":      round(cfg.typical_iv * 100, 2),
            "vrp_signal":          round(vrp_signal, 3),
            "liq_haircut":         liq_haircut,
            "raw_notional":        sizing.get("optimal_notional", 0),
            "adj_notional":        round(adj_notional, 0),
            "dollar_gamma_per_1pct": sizing.get("dollar_gamma_per_pct", 0),
            "theta_usd_day":       round(adj_budget, 0),
            "gamma_pnl_gross":     sizing.get("gamma_pnl_gross", 0),
            "net_pnl_est":         sizing.get("net_pnl_est", 0),
            "pnl_theta_ratio":     sizing.get("pnl_to_theta_ratio", 0),
            "squeeze_pips":        sq,
        })

    return pd.DataFrame(rows)

#  5: DYNAMIC GAMMA SIZING FROM HISTORICAL DATA

def load_price_data(ticker: str, period: str = "2y", interval: str = "1h") -> pd.Series:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
    close = df["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index, utc=True)
    return close


def rolling_rv(close: pd.Series, window: int = 24) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252 * 24)


def historical_gamma_sizing(
    key: str,
    theta_budget_usd: float = 1000,
    dte: float = 5,
    entry_pips: float = 50,
    entry_tol: float = 8,
) -> pd.DataFrame:
    """
    For each bar where entry conditions are met (spot ~entry_pips above round strike),
    compute the recommended gamma notional and log it alongside realized outcome.

    This lets you see: does higher recommended gamma (from regime model) predict
    better forward P&L? If yes, the sizing signal has value.
    """
    cfg   = ASSETS[key]
    T     = dte / 365
    print(f"  Loading {cfg.name} ({cfg.ticker})…")
    close = load_price_data(cfg.ticker)

    if close.empty or len(close) < 100:
        print(f"  No data for {key}")
        return pd.DataFrame()

    rv_series = rolling_rv(close, window=24)
    pip = cfg.pip_size
    rows = []

    for i in range(50, len(close) - 24):
        S = float(close.iloc[i])
        K = round(round(S / cfg.round_grid) * cfg.round_grid, 6)
        dist_pips = (S - K) / pip

        if not (entry_pips - entry_tol <= dist_pips <= entry_pips + entry_tol):
            continue

        rv_now = float(rv_series.iloc[i]) if not np.isnan(rv_series.iloc[i]) else cfg.typical_iv
        iv_now = max(rv_now * 1.15, 0.03)  # IV = RV × premium factor

        # Vol regime percentile (using available history up to this bar)
        rv_hist = rv_series.iloc[:i].dropna()
        if len(rv_hist) < 20:
            continue
        rv_pct = percentileofscore(rv_hist, rv_now)

        # Gamma multiplier from regime
        vrp = rv_now - iv_now
        if vrp > 0:
            gamma_mult = 1.0 + min(vrp / iv_now, 1.5)
        else:
            gamma_mult = max(1.0 + vrp / iv_now, 0.25)

        adj_budget = theta_budget_usd * gamma_mult
        sizing = optimal_notional_for_theta_budget(
            S, K, T, iv_now, entry_pips, adj_budget, cfg
        )

        # Forward realized P&L (simplified: half gamma × actual move²)
        fwd_bars  = min(24 * dte, len(close) - i - 1)
        S_fwd     = float(close.iloc[i + int(fwd_bars)])
        actual_move_pips = abs(S_fwd - K) / pip - dist_pips  # net move toward/away
        g = bs_gamma(S, K, T, iv_now)
        delta_s = (S_fwd - S)
        realized_gamma_pnl = 0.5 * g * sizing.get("optimal_notional", 0) * delta_s**2
        theta_cost = abs(bs_theta(S, K, T, iv_now)) * sizing.get("optimal_notional", 0) * dte
        net_pnl = realized_gamma_pnl - theta_cost

        rows.append({
            "date":             close.index[i],
            "spot":             round(S, 5),
            "strike":           round(K, 5),
            "dist_pips":        round(dist_pips, 1),
            "rv_pct":           round(rv_now * 100, 2),
            "iv_pct":           round(iv_now * 100, 2),
            "rv_regime_pctile": round(rv_pct, 1),
            "gamma_mult":       round(gamma_mult, 2),
            "adj_budget":       round(adj_budget, 0),
            "optimal_notional": sizing.get("optimal_notional", 0),
            "dollar_gamma":     sizing.get("dollar_gamma_per_pct", 0),
            "realized_gamma_pnl": round(realized_gamma_pnl, 0),
            "theta_cost":       round(theta_cost, 0),
            "net_pnl":          round(net_pnl, 0),
        })

    return pd.DataFrame(rows)

#  PLOTTING

def plot_gamma_surface(surface_df: pd.DataFrame, asset_name: str):
    pivot = surface_df.pivot_table(
        index="dist_pips", columns="iv", values="dollar_gamma_per_1pct"
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(
        pivot.values, cmap="YlOrRd", aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.max(), pivot.index.min()]
    )
    axes[0].set_xlabel("IV (%)", fontsize=9)
    axes[0].set_ylabel("Distance from strike (pips)", fontsize=9)
    axes[0].set_title(f"{asset_name} — dollar gamma per 1% spot move", fontsize=10)
    plt.colorbar(im, ax=axes[0], label="USD")

    # Edge exists heatmap
    pivot_edge = surface_df.pivot_table(
        index="dist_pips", columns="iv", values="iv_minus_min_rv"
    )
    im2 = axes[1].imshow(
        pivot_edge.values, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5,
        extent=[pivot_edge.columns.min(), pivot_edge.columns.max(),
                pivot_edge.index.max(), pivot_edge.index.min()]
    )
    axes[1].set_xlabel("IV (%)", fontsize=9)
    axes[1].set_ylabel("Distance from strike (pips)", fontsize=9)
    axes[1].set_title(f"{asset_name} — IV edge over min-RV breakeven (%)\nGreen = gamma profitable at current IV", fontsize=10)
    plt.colorbar(im2, ax=axes[1], label="IV − min RV (%)")

    plt.suptitle(f"Gamma sizing surface — {asset_name}", fontsize=12)
    fname = f"gamma_surface_{asset_name.replace('/', '').replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {fname}")


def plot_historical_sizing(hist_df: pd.DataFrame, asset_name: str):
    if hist_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Regime vs notional
    axes[0, 0].scatter(hist_df["rv_regime_pctile"], hist_df["optimal_notional"],
                       alpha=0.4, s=10, color="#378ADD")
    axes[0, 0].set_xlabel("RV regime percentile", fontsize=9)
    axes[0, 0].set_ylabel("Recommended notional (USD)", fontsize=9)
    axes[0, 0].set_title("Gamma sizing by vol regime", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Gamma multiplier distribution
    axes[0, 1].hist(hist_df["gamma_mult"], bins=25, color="#D85A30", alpha=0.8)
    axes[0, 1].axvline(1.0, color="gray", ls="--", lw=1)
    axes[0, 1].set_xlabel("Gamma multiplier", fontsize=9)
    axes[0, 1].set_title("Distribution of gamma multipliers\n(>1 = RV > IV, gamma cheap)", fontsize=10)

    # Net P&L vs gamma multiplier
    axes[1, 0].scatter(hist_df["gamma_mult"], hist_df["net_pnl"],
                       alpha=0.4, s=10, color="#1D9E75")
    axes[1, 0].axhline(0, color="gray", ls="--", lw=1)
    z = np.polyfit(hist_df["gamma_mult"].dropna(), hist_df["net_pnl"].dropna(), 1)
    xr = np.linspace(hist_df["gamma_mult"].min(), hist_df["gamma_mult"].max(), 50)
    axes[1, 0].plot(xr, np.poly1d(z)(xr), color="#D85A30", lw=1.5)
    axes[1, 0].set_xlabel("Gamma multiplier at entry", fontsize=9)
    axes[1, 0].set_ylabel("Net P&L (USD)", fontsize=9)
    axes[1, 0].set_title("Does higher gamma mult → better outcome?", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative P&L
    cum = hist_df.sort_values("date")["net_pnl"].cumsum()
    axes[1, 1].plot(range(len(cum)), cum.values, color="#378ADD", lw=1.5)
    axes[1, 1].axhline(0, color="gray", ls="--", lw=1)
    axes[1, 1].set_xlabel("Trade number", fontsize=9)
    axes[1, 1].set_ylabel("Cumulative P&L (USD)", fontsize=9)
    axes[1, 1].set_title("Cumulative P&L — regime-sized gamma", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Historical gamma sizing — {asset_name}", fontsize=12)
    fname = f"gamma_hist_{asset_name.replace('/', '').replace(' ','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {fname}")

if __name__ == "__main__":

    # ── Example spot/IV levels (update with live data) ──────────────────────
    SPOT_MAP = {
        "EURUSD": 1.0820,
        "USDJPY": 149.60,
        "GBPUSD": 1.2680,
        "SPX":    5250.0,
        "IBOV":   128_000.0,
    }
    IV_MAP = {
        "EURUSD": 0.073,
        "USDJPY": 0.082,
        "GBPUSD": 0.079,
        "SPX":    0.155,
        "IBOV":   0.220,
    }
    # Expected squeeze in pips toward nearest round strike
    SQUEEZE_MAP = {
        "EURUSD": 45,
        "USDJPY": 40,
        "GBPUSD": 40,
        "SPX":    60,    # 60 index points
        "IBOV":   800,   # 800 Bovespa points
    }
    THETA_BUDGET = 1_500   # USD per day max theta bleed

    print("=" * 60)
    print("  CROSS-ASSET GAMMA SIZING TABLE")
    print("=" * 60)
    table = cross_asset_gamma_table(THETA_BUDGET, SQUEEZE_MAP, SPOT_MAP, IV_MAP)
    print(table[[
        "asset", "class", "dist_pips", "iv_pct", "vrp_signal",
        "adj_notional", "dollar_gamma_per_1pct", "net_pnl_est", "pnl_theta_ratio"
    ]].to_string(index=False))

    table.to_csv("cross_asset_gamma_sizing.csv", index=False)
    print("\nSaved cross_asset_gamma_sizing.csv")

    print("\n" + "=" * 60)
    print("  GAMMA STRUCTURE COMPARISON — EURUSD")
    print("=" * 60)
    S, K = 1.0820, 1.0800
    structs = compare_gamma_structures(S, K, 5/365, 0.073, 1_000_000,
                                        ASSETS["EURUSD"])
    print(structs[["structure","dollar_gamma_per_1pct","theta_usd_day",
                   "prem_usd","gamma_theta_ratio","comment"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("  GAMMA SURFACE — EURUSD")
    print("=" * 60)
    surface = gamma_theta_breakeven_surface(
        ASSETS["EURUSD"], S=1.0820, T=5/365, notional=1_000_000
    )
    plot_gamma_surface(surface, "EUR/USD")

    print("\n" + "=" * 60)
    print("  HISTORICAL GAMMA SIZING — EURUSD & SPX")
    print("=" * 60)
    for key in ["EURUSD", "SPX"]:
        print(f"\n  {ASSETS[key].name}")
        hist = historical_gamma_sizing(key, theta_budget_usd=THETA_BUDGET)
        if not hist.empty:
            print(f"  Entries: {len(hist)}")
            print(f"  Mean gamma mult: {hist['gamma_mult'].mean():.2f}")
            print(f"  Net P&L (total): ${hist['net_pnl'].sum():,.0f}")
            print(f"  Win rate: {(hist['net_pnl']>0).mean():.1%}")
            hist.to_csv(f"gamma_hist_{key}.csv", index=False)
            plot_historical_sizing(hist, ASSETS[key].name)

    print("\nDone.")

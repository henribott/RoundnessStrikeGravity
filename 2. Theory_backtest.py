"""
Run after 01_data_loader.py has produced parquet files.
Also works standalone: fetches its own data if parquets not present.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind, pearsonr, spearmanr, mannwhitneyu
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
import warnings
warnings.filterwarnings("ignore")
@dataclass
class AssetConfig:
    name:        str
    asset_class: Literal["fx", "equity_index"]
    ticker:      str
    pip_size:    float
    round_grid:  float
    typical_iv:  float


ASSETS = {
    "EURUSD": AssetConfig("EUR/USD",  "fx",           "EURUSD=X", 0.0001, 0.0050, 0.07),
    "USDJPY": AssetConfig("USD/JPY",  "fx",           "USDJPY=X", 0.01,   0.50,   0.08),
    "GBPUSD": AssetConfig("GBP/USD",  "fx",           "GBPUSD=X", 0.0001, 0.0050, 0.08),
    "SPX":    AssetConfig("S&P 500",  "equity_index", "^GSPC",    1.0,    50.0,   0.17),
    "IBOV":   AssetConfig("Bovespa",  "equity_index", "^BVSP",    1.0,  1000.0,   0.22),
}

RV_WINDOW        = 24     # bars for realized vol (hourly data → 24h)
FWD_WINDOWS      = [6, 12, 24, 48]  # forward windows to test (bars)
ENTRY_PIPS       = 50     # spot this many pips above strike at entry
ENTRY_TOL        = 10     # ±tolerance
DTE_ENTRY        = 5      # days to expiry at entry
IV_PREMIUM       = 1.15   # IV = RV × this
THETA_BUDGET     = 1500   # USD/day
NOTIONAL         = 1_000_000
MIN_OBS          = 20     # minimum events for valid test
STRADDLE_NOTIONAL = 1_000_000

def load_or_fetch(key: str) -> pd.DataFrame:
    cfg  = ASSETS[key]
    path = Path(f"data_{key}.parquet")
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    print(f"  Fetching {cfg.name} from yfinance…")
    raw = yf.download(cfg.ticker, period="2y", interval="1h",
                      auto_adjust=True, progress=False)
    if raw.empty or len(raw) < 200:
        raw = yf.download(cfg.ticker, period="5y", interval="1d",
                          auto_adjust=True, progress=False)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index, utc=True)
    df = raw[["open", "high", "low", "close"]].dropna().copy()
    return df


def build_features(df: pd.DataFrame, cfg: AssetConfig) -> pd.DataFrame:
    """Add all derived columns needed for the four claims."""
    df = df.copy()
    pip = cfg.pip_size
    grid = cfg.round_grid

    # Nearest round strike and distance
    df["K_round"] = (df["close"] / grid).round() * grid
    df["dist_pips"] = (df["close"] - df["K_round"]) / pip

    # Roundness Ψ at nearest round strike (numerical salience)
    def _psi(K):
        levels = ([(0.1, 1.0), (0.05, 0.70), (0.025, 0.45),
                   (0.01, 0.25), (0.005, 0.12), (0.0025, 0.05)]
                  if cfg.asset_class == "fx" else
                  [(1000, 1.0), (500, 0.75), (250, 0.50),
                   (100, 0.30), (50, 0.15), (25, 0.07), (10, 0.03)])
        max_w = sum(w for _, w in levels)
        score = sum(w for d, w in levels if abs(K % d) < d * 1e-5
                    or abs(K % d - d) < d * 1e-5)
        return score / max_w

    df["psi"] = df["K_round"].apply(_psi)

    # Log returns and realized vol
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv"] = df["log_ret"].rolling(RV_WINDOW).std() * np.sqrt(252 * 24)

    # Implied vol estimate (RV × premium)
    df["iv_est"] = (df["rv"] * IV_PREMIUM).clip(lower=0.03)

    # Vol risk premium at bar (IV - RV; positive = IV expensive)
    df["vrp"] = df["iv_est"] - df["rv"]

    # Phi: ATM-normalized gamma proximity
    T = DTE_ENTRY / 365
    def _phi(row):
        S, K, sigma = row["close"], row["K_round"], max(row["iv_est"], 0.03)
        if T <= 0 or sigma <= 0:
            return 0.0
        d1_K   = (np.log(S / K)   + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d1_ATM = 0.5 * sigma * np.sqrt(T)  # log(S/S) = 0
        g_K    = norm.pdf(d1_K)   / (S * sigma * np.sqrt(T))
        g_ATM  = norm.pdf(d1_ATM) / (S * sigma * np.sqrt(T))
        return g_K / g_ATM if g_ATM > 1e-12 else 0.0

    df["phi"] = df.apply(_phi, axis=1)

    # Roundness scalar R = Ψ × Ω × Φ (Ω = Ψ^0.7 heuristic)
    df["omega"] = df["psi"] ** 0.7
    df["R"]     = df["psi"] * df["omega"] * df["phi"]

    # Forward realized vol for each window
    for w in FWD_WINDOWS:
        fwd_rv = (df["log_ret"].rolling(w).std() * np.sqrt(252 * 24)).shift(-w)
        df[f"fwd_rv_{w}"] = fwd_rv

    # Forward spot move (in pips)
    for w in FWD_WINDOWS:
        df[f"fwd_move_{w}"] = (df["close"].shift(-w) - df["close"]) / pip

    return df.dropna(subset=["rv", "iv_est", "R"])


#  CLAIM 1: Roundness predicts excess realized vol

def test_claim1_rv_excess(df: pd.DataFrame, cfg: AssetConfig) -> dict:
    """
    Split bars into HIGH-R (R > 0.5) and LOW-R (R < 0.2) groups
    within the same distance band from strike (±50p).
    Compare forward RV distributions.
    Hypothesis: HIGH-R bars have significantly larger forward RV.
    """
    results = {}
    nearby = df[df["dist_pips"].abs() <= 60].copy()

    high_R = nearby[nearby["R"] > 0.50]
    low_R  = nearby[nearby["R"] < 0.20]

    for w in FWD_WINDOWS:
        col = f"fwd_rv_{w}"
        h   = high_R[col].dropna()
        l   = low_R[col].dropna()
        if len(h) < MIN_OBS or len(l) < MIN_OBS:
            continue
        t_stat, p_val = ttest_ind(h, l, alternative="greater")
        mw_stat, mw_p = mannwhitneyu(h, l, alternative="greater")
        results[w] = {
            "n_high": len(h), "n_low": len(l),
            "mean_rv_high": round(h.mean() * 100, 3),
            "mean_rv_low":  round(l.mean() * 100, 3),
            "rv_excess_pct": round((h.mean() - l.mean()) * 100, 3),
            "t_stat": round(t_stat, 3),
            "p_ttest": round(p_val, 5),
            "p_mannwhitney": round(mw_p, 5),
            "supported": p_val < 0.05 and h.mean() > l.mean(),
        }

    return results


#  CLAIM 2: Edge ratio (RV > IV) near round strikes

def test_claim2_edge_ratio(df: pd.DataFrame, cfg: AssetConfig) -> dict:
    """
    Compute vol risk premium (VRP = IV - RV) in each zone.
    Near high-R strikes, VRP should be smaller (or negative):
    the market underprices vol because round-number OI creates
    realized vol that is not fully reflected in IV.

    Also runs a regression: VRP ~ R + dist_pips + rv_regime
    A negative coefficient on R means higher roundness → lower VRP → gamma cheap.
    """
    nearby = df[df["dist_pips"].abs() <= 100].copy()

    # VRP by R quartile
    nearby["R_quartile"] = pd.qcut(nearby["R"], 4,
                                    labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    vrp_by_q = nearby.groupby("R_quartile", observed=True)["vrp"].agg(
        ["mean", "median", "std", "count"]
    ).round(5)

    # Regression: VRP ~ R
    from scipy.stats import linregress
    mask = nearby["R"].notna() & nearby["vrp"].notna()
    slope, intercept, r_val, p_val, std_err = linregress(
        nearby.loc[mask, "R"], nearby.loc[mask, "vrp"]
    )

    # Pearson correlation R ↔ VRP
    corr, corr_p = pearsonr(
        nearby.loc[mask, "R"], nearby.loc[mask, "vrp"]
    )

    return {
        "vrp_by_R_quartile": vrp_by_q,
        "regression_slope":   round(slope, 6),
        "regression_p":       round(p_val, 5),
        "pearson_corr":       round(corr, 4),
        "pearson_p":          round(corr_p, 5),
        "supported": slope < 0 and p_val < 0.05,
        "interpretation": (
            "Higher R → lower VRP → gamma cheap near round strikes"
            if slope < 0 else
            "No evidence that gamma is cheaper near round strikes"
        ),
    }


#  CLAIM 3: GEX cumulative integral predicts squeeze magnitude

def gex_integral(S: float, K: float, T: float, sigma: float,
                 oi_weight: float = 1.0, pip: float = 0.0001) -> float:
    """
    Cumulative GEX from current spot S to strike K.
    GEX_cumulative = ∫_min(S,K)^max(S,K)  Γ(x, K, t, σ) × OI_weight  dx

    This measures total dealer hedging obligation as spot traverses S→K.
    Units: gamma × price units → dimensionless integral (normalized by S later)
    """
    if abs(S - K) < pip * 0.1:
        return 0.0

    def integrand(x):
        if T <= 1e-8 or sigma <= 1e-8:
            return 0.0
        d1 = (np.log(x / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (x * sigma * np.sqrt(T)) * oi_weight

    lo, hi = min(S, K), max(S, K)
    result, _ = quad(integrand, lo, hi, limit=50)
    return result


def test_claim3_gex_cascade(df: pd.DataFrame, cfg: AssetConfig) -> dict:
    """
    For each entry bar (spot within ENTRY_PIPS ± TOL of round strike),
    compute GEX integral and compare to:
      (a) actual forward spot move toward strike
      (b) whether entry spot eventually touched the strike within FWD_WINDOWS

    Hypothesis: GEX_integral > threshold predicts stronger squeeze than
    distance alone.
    """
    pip = cfg.pip_size
    T   = DTE_ENTRY / 365
    rows = []

    # Entry condition
    entries = df[
        (df["dist_pips"].abs().between(ENTRY_PIPS - ENTRY_TOL,
                                        ENTRY_PIPS + ENTRY_TOL)) &
        (df["R"] > 0.1)
    ].copy()

    if len(entries) < MIN_OBS:
        return {"n": len(entries), "valid": False,
                "reason": "insufficient entries"}

    for idx, row in entries.iterrows():
        S     = row["close"]
        K     = row["K_round"]
        sigma = max(row["iv_est"], 0.03)
        omega = row["omega"]

        gex = gex_integral(S, K, T, sigma, oi_weight=omega, pip=pip)

        # Forward move toward strike (negative dist_pips = moved toward K)
        fwd_24 = row.get("fwd_move_24", np.nan)

        rows.append({
            "gex_integral":  gex,
            "dist_pips":     row["dist_pips"],
            "R":             row["R"],
            "psi":           row["psi"],
            "fwd_move_24":   fwd_24,
            "toward_strike": -fwd_24 if not np.isnan(fwd_24) else np.nan,
        })

    gex_df = pd.DataFrame(rows).dropna()
    if len(gex_df) < MIN_OBS:
        return {"n": len(gex_df), "valid": False,
                "reason": "insufficient valid entries"}

    # Correlation: GEX integral ↔ move toward strike
    corr_gex, p_gex = spearmanr(gex_df["gex_integral"],
                                  gex_df["toward_strike"])
    corr_dist, p_dist = spearmanr(gex_df["dist_pips"].abs(),
                                   gex_df["toward_strike"])
    corr_R, p_R = spearmanr(gex_df["R"], gex_df["toward_strike"])

    # Partial correlation: does GEX add info beyond distance alone?
    gex_df["gex_resid"] = (
        gex_df["gex_integral"] -
        np.polyval(np.polyfit(gex_df["dist_pips"].abs(),
                              gex_df["gex_integral"], 1),
                   gex_df["dist_pips"].abs())
    )
    partial_corr, partial_p = spearmanr(gex_df["gex_resid"],
                                         gex_df["toward_strike"])

    return {
        "n": len(gex_df),
        "valid": True,
        "corr_gex_vs_move":    round(corr_gex, 4),
        "p_gex_vs_move":       round(p_gex, 5),
        "corr_dist_vs_move":   round(corr_dist, 4),
        "p_dist_vs_move":      round(p_dist, 5),
        "corr_R_vs_move":      round(corr_R, 4),
        "p_R_vs_move":         round(p_R, 5),
        "partial_corr_gex":    round(partial_corr, 4),
        "partial_p_gex":       round(partial_p, 5),
        "gex_adds_info":       partial_p < 0.05,
        "supported": corr_gex > 0 and p_gex < 0.05,
        "gex_df": gex_df,
    }


#  CLAIM 4: R-based strategy selection produces positive expected value

def bs_straddle_pnl(S_entry, K, T_entry, sigma_iv, S_exit, T_exit):
    """Straddle P&L: mark to market using same IV (conservative)."""
    def call(S, T):
        if T <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + 0.5 * sigma_iv**2 * T) / (sigma_iv * np.sqrt(T))
        d2 = d1 - sigma_iv * np.sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)

    def put(S, T):
        if T <= 0:
            return max(K - S, 0)
        d1 = (np.log(S / K) + 0.5 * sigma_iv**2 * T) / (sigma_iv * np.sqrt(T))
        d2 = d1 - sigma_iv * np.sqrt(T)
        return K * norm.cdf(-d2) - S * norm.cdf(-d1)

    prem_entry = call(S_entry, T_entry) + put(S_entry, T_entry)
    prem_exit  = call(S_exit,  T_exit)  + put(S_exit,  T_exit)
    return prem_exit - prem_entry


def select_structure(R: float) -> str:
    if R < 0.30:   return "skip"
    elif R < 0.55: return "strangle"
    elif R < 0.75: return "atm_straddle"
    else:          return "ratio_1x2"


def test_claim4_strategy_pnl(df: pd.DataFrame, cfg: AssetConfig) -> pd.DataFrame:
    """
    Systematic backtest: at each valid entry, select structure by R,
    hold for min(DTE, until TP/SL hit), record P&L.

    Returns trade log DataFrame.
    """
    pip = cfg.pip_size
    T0  = DTE_ENTRY / 365
    trades = []
    last_exit = -1
    cooldown  = 12  # bars

    for i in range(RV_WINDOW, len(df) - 48):
        if i - last_exit < cooldown:
            continue

        row = df.iloc[i]
        dp  = row["dist_pips"]

        if not (ENTRY_PIPS - ENTRY_TOL <= abs(dp) <= ENTRY_PIPS + ENTRY_TOL):
            continue

        R       = row["R"]
        strat   = select_structure(R)
        if strat == "skip":
            continue

        S       = row["close"]
        K       = row["K_round"]
        sigma   = max(row["iv_est"], 0.03)

        # For strangle: widen strikes by 50% of entry distance
        K_call  = K + abs(dp) * 0.5 * pip
        K_put   = K - abs(dp) * 0.5 * pip

        # Size by R: higher R → larger notional (up to 2x base)
        size_mult = 0.5 + R  # R=0.3→0.8x, R=0.75→1.25x, R=1.0→1.5x
        notional  = STRADDLE_NOTIONAL * size_mult

        # TP / SL in pips
        tp_pips = 35
        sl_pips = -20
        max_hold = min(DTE_ENTRY * 24, 48)

        exit_bar   = None
        exit_pnl   = None
        exit_reason = "expiry"

        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            bars_held = j - i
            T_now = max(T0 - bars_held / (252 * 24), 1e-6)
            S_now = df.iloc[j]["close"]

            if strat == "atm_straddle":
                pnl_price = bs_straddle_pnl(S, K, T0, sigma, S_now, T_now)
            elif strat == "strangle":
                pnl_c = bs_straddle_pnl(S, K_call, T0, sigma, S_now, T_now)
                pnl_p = bs_straddle_pnl(S, K_put,  T0, sigma, S_now, T_now)
                pnl_price = pnl_c + pnl_p
            else:  # ratio_1x2: long 1 ATM call, short 2 OTM calls
                pnl_long  = bs_straddle_pnl(S, K, T0, sigma, S_now, T_now) * 0.5
                K_otm     = K + abs(dp) * pip
                def _call(spot, t):
                    if t <= 0: return max(spot - K_otm, 0)
                    d1 = (np.log(spot/K_otm) + 0.5*sigma**2*t)/(sigma*np.sqrt(t))
                    return spot*norm.cdf(d1) - K_otm*norm.cdf(d1-sigma*np.sqrt(t))
                pnl_short = _call(S_now, T_now) - _call(S, T0)
                pnl_price = pnl_long - 2 * pnl_short

            pnl_pips = pnl_price / pip

            if pnl_pips >= tp_pips:
                exit_bar, exit_pnl, exit_reason = j, pnl_pips, "take_profit"
                break
            elif pnl_pips <= sl_pips:
                exit_bar, exit_pnl, exit_reason = j, pnl_pips, "stop_loss"
                break

        if exit_bar is None:
            exit_bar   = min(i + max_hold, len(df) - 1)
            T_exit     = max(T0 - max_hold / (252 * 24), 1e-6)
            S_exit     = df.iloc[exit_bar]["close"]
            pnl_price  = bs_straddle_pnl(S, K, T0, sigma, S_exit, T_exit)
            exit_pnl   = pnl_price / pip

        last_exit = exit_bar

        trades.append({
            "entry_idx":   i,
            "exit_idx":    exit_bar,
            "date":        df.index[i],
            "strike":      round(K, 5),
            "spot":        round(S, 5),
            "dist_pips":   round(dp, 1),
            "R":           round(R, 4),
            "psi":         round(row["psi"], 4),
            "phi":         round(row["phi"], 4),
            "iv_est_pct":  round(sigma * 100, 2),
            "rv_pct":      round(row["rv"] * 100, 2),
            "vrp_pct":     round(row["vrp"] * 100, 3),
            "strategy":    strat,
            "size_mult":   round(size_mult, 2),
            "notional":    round(notional, 0),
            "pnl_pips":    round(exit_pnl, 2),
            "pnl_usd":     round(exit_pnl * pip * notional, 0),
            "exit_reason": exit_reason,
            "bars_held":   exit_bar - i,
        })

    return pd.DataFrame(trades)

#  REFLEXIVE VOL: does the approach to a round number create a vol spike?
#  This tests the "reflex vol" concept: spot moving TOWARD K causes IV to rise.
#  Measured as: IV_{t+k} - IV_t  conditional on spot moving toward K.

def test_reflex_vol(df: pd.DataFrame, cfg: AssetConfig) -> dict:
    """
    For bars near high-R strikes where spot moves TOWARD K in the next 6h,
    compare IV change vs bars where spot moves AWAY.
    Hypothesis: approach to round strike elevates estimated IV (reflex vol).
    """
    nearby = df[df["dist_pips"].abs().between(5, 80)].copy()
    nearby["iv_change_6"] = (nearby["iv_est"].shift(-6) -
                              nearby["iv_est"])
    nearby["moved_toward"] = (
        (nearby["dist_pips"] > 0) & (nearby["fwd_move_6"] < 0) |
        (nearby["dist_pips"] < 0) & (nearby["fwd_move_6"] > 0)
    )
    nearby = nearby.dropna(subset=["iv_change_6", "moved_toward"])

    toward  = nearby[nearby["moved_toward"] & (nearby["R"] > 0.4)]["iv_change_6"]
    away    = nearby[~nearby["moved_toward"] & (nearby["R"] > 0.4)]["iv_change_6"]

    if len(toward) < MIN_OBS or len(away) < MIN_OBS:
        return {"valid": False, "n_toward": len(toward), "n_away": len(away)}

    t_stat, p_val = ttest_ind(toward, away, alternative="greater")
    return {
        "valid": True,
        "n_toward": len(toward), "n_away": len(away),
        "mean_iv_change_toward": round(toward.mean() * 100, 4),
        "mean_iv_change_away":   round(away.mean() * 100, 4),
        "t_stat": round(t_stat, 3),
        "p_val":  round(p_val, 5),
        "supported": p_val < 0.05 and toward.mean() > away.mean(),
    }

# Plotting

def plot_claim1(c1: dict, pair: str, df: pd.DataFrame):
    windows  = [w for w in FWD_WINDOWS if w in c1]
    if not windows: return

    high_rvs = [c1[w]["mean_rv_high"] for w in windows]
    low_rvs  = [c1[w]["mean_rv_low"]  for w in windows]
    p_vals   = [c1[w]["p_ttest"]      for w in windows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(windows))
    axes[0].bar(x - 0.2, high_rvs, 0.35, label="High-R (round strike)", color="#D85A30", alpha=0.85)
    axes[0].bar(x + 0.2, low_rvs,  0.35, label="Low-R (non-round)",     color="#378ADD", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{w}h fwd" for w in windows])
    axes[0].set_ylabel("Mean realized vol (%)")
    axes[0].set_title(f"Claim 1: RV near round strikes > non-round\n{pair}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    colors = ["#1D9E75" if p < 0.05 else "#D85A30" for p in p_vals]
    axes[1].bar(x, p_vals, 0.5, color=colors, alpha=0.85)
    axes[1].axhline(0.05, color="gray", ls="--", lw=1, label="p=0.05")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{w}h" for w in windows])
    axes[1].set_ylabel("p-value (t-test)")
    axes[1].set_title("p-values (green = significant)")
    axes[1].legend()

    plt.suptitle(f"Claim 1 — {pair}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"claim1_{pair}.png", dpi=130, bbox_inches="tight")
    plt.show()


def plot_claim2(c2: dict, pair: str, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # VRP by R quartile
    vrp_q = c2["vrp_by_R_quartile"]
    axes[0].bar(vrp_q.index.astype(str), vrp_q["mean"] * 100,
                color=["#B4B2A9", "#378ADD", "#EF9F27", "#D85A30"], alpha=0.85)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_ylabel("Mean VRP (IV - RV) %")
    axes[0].set_title(f"Claim 2: VRP by R quartile\n{pair}\nSlope={c2['regression_slope']:.5f}  p={c2['regression_p']:.4f}")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Scatter R vs VRP
    nearby = df[df["dist_pips"].abs() <= 100].sample(min(2000, len(df)))
    axes[1].scatter(nearby["R"], nearby["vrp"] * 100,
                    alpha=0.2, s=8, color="#7F77DD")
    x_range = np.linspace(nearby["R"].min(), nearby["R"].max(), 50)
    slope    = c2["regression_slope"]
    intercept = nearby["vrp"].mean() * 100  # rough intercept
    axes[1].plot(x_range, slope * 1000 * x_range + intercept,
                 color="#D85A30", lw=1.5,
                 label=f"Regression  ρ={c2['pearson_corr']:.3f}")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("R (roundness)")
    axes[1].set_ylabel("VRP (IV - RV) %")
    axes[1].set_title("R vs vol risk premium")
    axes[1].legend()

    plt.suptitle(f"Claim 2 — {pair}  "
                 f"{'SUPPORTED' if c2['supported'] else 'NOT SUPPORTED'}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"claim2_{pair}.png", dpi=130, bbox_inches="tight")
    plt.show()


def plot_claim3(c3: dict, pair: str):
    if not c3.get("valid"):
        print(f"  Claim 3 ({pair}): insufficient data")
        return

    gex_df = c3["gex_df"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(gex_df["gex_integral"], gex_df["toward_strike"],
                    alpha=0.4, s=15, color="#D85A30")
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_xlabel("GEX cumulative integral")
    axes[0].set_ylabel("Move toward strike (pips)")
    axes[0].set_title(f"Claim 3: GEX integral vs squeeze\n"
                       f"ρ={c3['corr_gex_vs_move']:.3f}  p={c3['p_gex_vs_move']:.4f}")
    axes[0].grid(True, alpha=0.3)

    # Compare: GEX vs distance as predictors
    predictors = ["GEX integral", "Distance alone", "R scalar"]
    corrs      = [abs(c3["corr_gex_vs_move"]),
                  abs(c3["corr_dist_vs_move"]),
                  abs(c3["corr_R_vs_move"])]
    ps         = [c3["p_gex_vs_move"], c3["p_dist_vs_move"], c3["p_R_vs_move"]]
    colors     = ["#1D9E75" if p < 0.05 else "#B4B2A9" for p in ps]
    axes[1].bar(predictors, corrs, color=colors, alpha=0.85)
    axes[1].set_ylabel("|Spearman ρ| with move toward strike")
    axes[1].set_title(f"Which predictor is stronger?\nPartial GEX ρ={c3['partial_corr_gex']:.3f}  p={c3['partial_p_gex']:.4f}")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Claim 3 — {pair}  "
                 f"{'GEX ADDS INFO' if c3['gex_adds_info'] else 'GEX no extra info'}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"claim3_{pair}.png", dpi=130, bbox_inches="tight")
    plt.show()


def plot_claim4(trades: pd.DataFrame, pair: str):
    if trades.empty:
        print(f"  Claim 4 ({pair}): no trades")
        return

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Cumulative P&L
    ax1 = fig.add_subplot(gs[0, :2])
    strat_colors = {"strangle": "#378ADD", "atm_straddle": "#EF9F27", "ratio_1x2": "#D85A30"}
    for strat, grp in trades.groupby("strategy"):
        cum = grp.sort_values("date")["pnl_usd"].cumsum()
        ax1.plot(range(len(cum)), cum.values,
                 color=strat_colors.get(strat, "#888"), label=strat, lw=1.5)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_title(f"Cumulative P&L by strategy — {pair}")
    ax1.set_ylabel("USD")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # P&L distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(trades["pnl_pips"], bins=30, color="#7F77DD", alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="gray", lw=1, ls="--")
    ax2.axvline(trades["pnl_pips"].mean(), color="#D85A30", lw=1.5,
                label=f"Mean {trades['pnl_pips'].mean():.1f}p")
    ax2.set_title("P&L distribution (pips)")
    ax2.legend()

    # R vs P&L scatter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(trades["R"], trades["pnl_pips"], alpha=0.4, s=15, color="#378ADD")
    ax3.axhline(0, color="gray", lw=0.8, ls="--")
    corr_r, p_r = pearsonr(trades["R"], trades["pnl_pips"])
    ax3.set_xlabel("R at entry")
    ax3.set_ylabel("P&L (pips)")
    ax3.set_title(f"R vs outcome\nρ={corr_r:.3f}  p={p_r:.4f}")
    ax3.grid(True, alpha=0.3)

    # VRP vs P&L
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(trades["vrp_pct"], trades["pnl_pips"], alpha=0.4, s=15, color="#D85A30")
    ax4.axhline(0, color="gray", lw=0.8, ls="--")
    ax4.axvline(0, color="gray", lw=0.8, ls="--")
    corr_v, p_v = pearsonr(trades["vrp_pct"], trades["pnl_pips"])
    ax4.set_xlabel("VRP at entry (IV-RV %)")
    ax4.set_ylabel("P&L (pips)")
    ax4.set_title(f"VRP vs outcome\nρ={corr_v:.3f}  p={p_v:.4f}")
    ax4.grid(True, alpha=0.3)

    # Exit reason
    ax5 = fig.add_subplot(gs[1, 2])
    ec = trades["exit_reason"].value_counts()
    ec_colors = [{"take_profit":"#1D9E75","stop_loss":"#D85A30",
                   "expiry":"#888780"}.get(e,"#888") for e in ec.index]
    ax5.bar(ec.index, ec.values, color=ec_colors, alpha=0.85)
    ax5.set_title("Exit reasons")
    for b in ax5.patches:
        ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 str(int(b.get_height())), ha="center", fontsize=9)

    win_rate = (trades["pnl_pips"] > 0).mean()
    total_pnl = trades["pnl_usd"].sum()
    plt.suptitle(f"Claim 4 — {pair}  |  Trades: {len(trades)}  "
                 f"Win: {win_rate:.0%}  Total: ${total_pnl:,.0f}", fontsize=11)
    plt.savefig(f"claim4_{pair}.png", dpi=130, bbox_inches="tight")
    plt.show()


# SUMMARY VERDICT TABLE

def print_verdict(pair: str, c1: dict, c2: dict, c3: dict,
                  trades: pd.DataFrame, reflex: dict):
    def support(x): return "✓ SUPPORTED" if x else "✗ NOT SUPPORTED"

    c1_sup = any(c1.get(w, {}).get("supported", False) for w in FWD_WINDOWS)
    c4_sup = not trades.empty and trades["pnl_pips"].mean() > 0

    print(f"\n{'═'*58}")
    print(f"  VERDICT — {pair}")
    print(f"{'═'*58}")
    print(f"  Claim 1 (RV excess near round strikes):  {support(c1_sup)}")
    print(f"  Claim 2 (gamma cheap near round strikes):{support(c2.get('supported', False))}")
    print(f"  Claim 3 (GEX integral predicts squeeze): {support(c3.get('supported', False))}")
    print(f"  Claim 4 (R-strategy → positive EV):     {support(c4_sup)}")
    print(f"  Reflex vol (approach elevates IV):       {support(reflex.get('supported', False))}")
    if not trades.empty:
        print(f"\n  Backtest:  {len(trades)} trades  |  "
              f"Win {(trades['pnl_pips']>0).mean():.0%}  |  "
              f"${trades['pnl_usd'].sum():,.0f} total  |  "
              f"Mean {trades['pnl_pips'].mean():.1f}p/trade")
    print(f"{'═'*58}")


if __name__ == "__main__":
    TEST_ASSETS = ["EURUSD", "USDJPY", "SPX"]  # add IBOV, GBPUSD as desired
    all_trades  = []

    for key in TEST_ASSETS:
        cfg = ASSETS[key]
        print(f"\n{'─'*58}")
        print(f"  Testing {cfg.name}")
        print(f"{'─'*58}")

        raw = load_or_fetch(key)
        df  = build_features(raw, cfg)

        # need fwd_move_6 for reflex vol test
        pip = cfg.pip_size
        df["fwd_move_6"] = (df["close"].shift(-6) - df["close"]) / pip

        print(f"  {len(df):,} bars  |  R mean={df['R'].mean():.4f}  "
              f"R max={df['R'].max():.4f}")

        c1     = test_claim1_rv_excess(df, cfg)
        c2     = test_claim2_edge_ratio(df, cfg)
        c3     = test_claim3_gex_cascade(df, cfg)
        trades = test_claim4_strategy_pnl(df, cfg)
        reflex = test_reflex_vol(df, cfg)

        all_trades.append(trades)

        print_verdict(key, c1, c2, c3, trades, reflex)

        plot_claim1(c1, cfg.name, df)
        plot_claim2(c2, cfg.name, df)
        plot_claim3(c3, cfg.name)
        plot_claim4(trades, cfg.name)

        trades.to_csv(f"backtest_trades_{key}.csv", index=False)

    # Combined summary
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv("backtest_all_trades.csv", index=False)
        print(f"\n  All trades saved: {len(combined)} total")
        print(f"  Combined win rate: {(combined['pnl_pips']>0).mean():.1%}")
        print(f"  Combined P&L:      ${combined['pnl_usd'].sum():,.0f}")

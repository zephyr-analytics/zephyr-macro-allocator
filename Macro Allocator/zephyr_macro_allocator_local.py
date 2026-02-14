import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# CONFIG (QC IDENTICAL)
# ============================
CRYPTO_CAP = 0.10
ENABLE_SMA_FILTER = True
ENABLE_TREASURY_KILL_SWITCH = True
USE_GROUP_MOMENTUM = False 

WINRATE_LOOKBACK = 126
VOL_LOOKBACK = 126

SMA_PERIOD = 147
BOND_SMA_PERIOD = 126

MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)

# ============================
# ASSET GROUPS
# ============================
GROUPS = {
    "real": ["GLD", "PDBC"],
    "corp_bonds": ["VCSH", "VCIT", "VCLT"],
    "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
    "high_yield_bonds": ["SHYG", "HYG"],
    "equities": ["VTI", "VEA", "VWO"],
    "equity_income": ["VIG", "VYM", "VIGI", "VYMI"],
    "crypto": ["IBIT", "ETHA"], 
    "cash": ["SHV"]
}

BOND_GROUPS = {"corp_bonds", "treasury_bonds", "high_yield_bonds"}

# ============================
# DATA FETCHING
# ============================
tickers = sorted(set(sum(GROUPS.values(), [])))
raw_data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)

closes = raw_data["Close"].dropna(how="all")
highs = raw_data["High"]
lows = raw_data["Low"]

# ============================
# CORE FUNCTIONS
# ============================

def compute_manual_adx(ticker, period=14):
    h, l, c = highs[ticker], lows[ticker], closes[ticker]
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    alpha = 1 / period
    s_plus_dm = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    s_minus_dm = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
    s_tr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (s_plus_dm / s_tr)
    minus_di = 100 * (s_minus_dm / s_tr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

def passes_trend(t):
    if not ENABLE_SMA_FILTER: return True
    px = closes[t]
    period = BOND_SMA_PERIOD if any(t in GROUPS[g] for g in BOND_GROUPS) else SMA_PERIOD
    ma = px.rolling(period).mean()
    return px.iloc[-1] > ma.iloc[-1]

def momentum(t):
    px = closes[t]
    return np.mean([px.iloc[-1] / px.iloc[-(lb + 1)] - 1 for lb in MOMENTUM_LOOKBACKS])

def duration_regime(symbols):
    def m(t):
        px = closes[t]
        return np.mean([px.iloc[-1]/px.iloc[-22]-1, px.iloc[-1]/px.iloc[-64]-1, px.iloc[-1]/px.iloc[-127]-1])
    valid = [s for s in symbols if s in closes.columns]
    if len(valid) == 3:
        s, i, l = valid
        return [s, i, l] if m(l) > m(i) > m(s) else [s, i] if m(i) > m(s) else [s]
    return valid

# ============================
# EXECUTION
# ============================

# 1. Treasury Kill Switch
if ENABLE_TREASURY_KILL_SWITCH and all(not passes_trend(t) for t in GROUPS["treasury_bonds"]):
    print("TREASURY KILL SWITCH: all treasuries failed trend → 100% CASH")
    pd.Series({"SHV": 1.0}).to_csv("signals.csv")
    exit()

bil_6m = closes["SHV"].iloc[-1] / closes["SHV"].iloc[-127] - 1
risk_groups = {g: (duration_regime(GROUPS[g]) if g in BOND_GROUPS else GROUPS[g]) for g in GROUPS if g != "cash"}

edges, group_assets, group_asset_edges = {}, {}, {}

for group, symbols in risk_groups.items():
    eligible, asset_edges, asset_adxs = [], {}, {}

    for s in symbols:
        if not passes_trend(s): continue
        m_6m = closes[s].iloc[-1] / closes[s].iloc[-127] - 1
        mom = momentum(s)
        if m_6m < bil_6m or mom <= 0: continue
        
        eligible.append(s)
        asset_edges[s] = mom
        asset_adxs[s] = compute_manual_adx(s)

    if not eligible: continue

    edge_series = pd.Series(asset_edges)
    asset_weights = edge_series / edge_series.sum()
    weighted_group_rets = (closes[eligible].pct_change() * asset_weights).sum(axis=1).dropna()
    log_group = np.log1p(weighted_group_rets)

    win_rate = (log_group.tail(WINRATE_LOOKBACK) > 0).mean()
    group_vol = np.std(log_group.tail(VOL_LOOKBACK)) * np.sqrt(252)
    group_adx = (pd.Series(asset_adxs) * asset_weights).sum()
    group_momentum = (edge_series * asset_weights).sum()

    if group_vol <= 0: continue

    confidence = (group_momentum * group_adx) / (group_vol + 1e-6)
    edges[group] = group_momentum * group_adx if USE_GROUP_MOMENTUM else win_rate * (1.0 + confidence)
    group_assets[group], group_asset_edges[group] = eligible, asset_edges

# ============================
# FINAL ALLOCATION
# ============================
if not edges:
    weights, cash_weight, alloc = {}, 1.0, {"SHV": 1.0}
else:
    total_edge = sum(edges.values())
    weights = {g: (edges[g] / total_edge) for g in edges}

    # Crypto Cap
    if "crypto" in weights and weights["crypto"] > CRYPTO_CAP:
        excess = weights["crypto"] - CRYPTO_CAP
        weights["crypto"] = CRYPTO_CAP
        others = [g for g in weights if g != "crypto"]
        total_other = sum(weights[g] for g in others)
        for g in others: weights[g] += excess * (weights[g] / total_other)

    cash_weight = max(0.0, 1.0 - sum(weights.values()))
    alloc = {}
    for g, w in weights.items():
        g_edges = pd.Series(group_asset_edges[g])
        ssum = g_edges.sum()
        for s, e in g_edges.items():
            alloc[s] = w * (e / ssum)
    alloc["SHV"] = cash_weight

# Summary Prints
print("GROUPS | " + " | ".join(f"{g}:{w:.2f}" for g, w in sorted(weights.items())) + f" | cash:{cash_weight:.2f}")
print("\nFINAL SIGNALS (%)\n")
out = pd.Series(alloc).sort_values(ascending=False)
print((out * 100).round(2))

import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# CONFIG (QC IDENTICAL)
# ============================
CRYPTO_CAP = 0.10

ENABLE_SMA_FILTER = True
ENABLE_TREASURY_KILL_SWITCH = True
ENABLE_GROUP_MOMENTUM = False

WINRATE_LOOKBACK = 126
VOL_LOOKBACK = 126

SMA_PERIOD = 147
BOND_SMA_PERIOD = 126

MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)


# ============================
# ASSET GROUPS (QC MATCH)
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
# DATA
# ============================
tickers = sorted(set(sum(GROUPS.values(), [])))

data = yf.download(
    tickers,
    period=f"{MAX_LOOKBACK + 50}d",
    auto_adjust=True,
    progress=False
)["Close"].dropna(how="all")

# ============================
# INDICATORS
# ============================
def sma(px, n):
    return px.rolling(n).mean()

def passes_trend(t):
    if not ENABLE_SMA_FILTER:
        return True

    px = data[t]
    ma = sma(
        px,
        BOND_SMA_PERIOD if any(t in GROUPS[g] for g in BOND_GROUPS)
        else SMA_PERIOD
    )
    return ma.notna().iloc[-1] and px.iloc[-1] > ma.iloc[-1]

def momentum(t):
    px = data[t]
    return np.mean([
        px.iloc[-1] / px.iloc[-(lb + 1)] - 1
        for lb in MOMENTUM_LOOKBACKS
    ])

def six_month_return(t):
    px = data[t]
    return px.iloc[-1] / px.iloc[-127] - 1

def duration_regime(symbols):
    def m(t):
        px = data[t]
        return np.mean([
            px.iloc[-1]/px.iloc[-22]-1,
            px.iloc[-1]/px.iloc[-64]-1,
            px.iloc[-1]/px.iloc[-127]-1,
        ])

    if len(symbols) == 3:
        s, i, l = symbols
        return (
            [s, i, l] if m(l) > m(i) > m(s)
            else [s, i] if m(i) > m(s)
            else [s]
        )
    if len(symbols) == 2:
        s, l = symbols
        return [s, l] if m(l) > m(s) else [s]
    return symbols

# ============================
# BUILD BASE RISK GROUPS
# ============================
risk_groups = {
    "real": GROUPS["real"],
    "corp_bonds": duration_regime(GROUPS["corp_bonds"]),
    "treasury_bonds": duration_regime(GROUPS["treasury_bonds"]),
    "high_yield_bonds": duration_regime(GROUPS["high_yield_bonds"]),
    "equities": GROUPS["equities"],
    "equity_income": GROUPS["equity_income"],
    "crypto": GROUPS["crypto"]
}

# ============================
# TREASURY KILL SWITCH (QC)
# ============================
if ENABLE_TREASURY_KILL_SWITCH:
    if all(not passes_trend(t) for t in GROUPS["treasury_bonds"]):
        print("TREASURY KILL SWITCH → 100% CASH")
        pd.Series({"SHV": 1.0}).to_csv("signals.csv")
        raise SystemExit

# ============================
# EDGE CONSTRUCTION (QC)
# ============================
edges, vols = {}, {}
group_assets, group_asset_edges = {}, {}

bil_6m = six_month_return("SHV")

for group, symbols in risk_groups.items():
    eligible, asset_edges = [], {}

    for s in symbols:
        if not passes_trend(s):
            continue
        if six_month_return(s) < bil_6m:
            continue

        mom = momentum(s)
        if mom <= 0:
            continue

        asset_edges[s] = mom
        eligible.append(s)

    if not eligible:
        continue

    group_assets[group] = eligible
    group_asset_edges[group] = asset_edges

    rets = data[eligible].pct_change().mean(axis=1).dropna()
    log_rets = np.log1p(rets)

    win_rate = (log_rets.tail(WINRATE_LOOKBACK) > 0).mean()
    vol = np.std(log_rets.tail(VOL_LOOKBACK)) * np.sqrt(252)

    group_mom = np.mean([momentum(s) for s in eligible])

    if not np.isfinite(group_mom) or group_mom <=0:
        continue

    confidence = group_mom / (vol + 1e-6)

    if ENABLE_GROUP_MOMENTUM:
        edges[group] = group_mom
    else:
        edges[group] = win_rate * (1.0 + confidence)

    vols[group] = vol

# ============================
# GROUP WEIGHTS
# ============================
eff = {g: e for g, e in edges.items()}
total = sum(eff.values())

weights = {
    g: (eff[g] / total)
    for g in eff
}

# ============================
# CRYPTO CAP (QC)
# ============================
if "crypto" in weights and weights["crypto"] > CRYPTO_CAP:
    excess = weights["crypto"] - CRYPTO_CAP
    weights["crypto"] = CRYPTO_CAP
    others = [g for g in weights if g != "crypto"]
    total_other = sum(weights[g] for g in others)
    for g in others:
        weights[g] += excess * (weights[g] / total_other)

cash_weight = max(0.0, 1.0 - sum(weights.values()))

# ============================
# FINAL ALLOCATION
# ============================
alloc = {}

for g, w in weights.items():
    edges_i = {s: group_asset_edges[g][s] for s in group_assets[g]}
    ssum = sum(edges_i.values())
    for s, e in edges_i.items():
        alloc[s] = w * e / ssum

alloc["SHV"] = cash_weight

out = pd.Series(alloc).sort_values(ascending=False)

print("GROUPS | " + " | ".join(f"{g}:{w:.2f}" for g, w in sorted(weights.items())) + f" | cash:{cash_weight:.2f}")
print("\nFINAL SIGNALS (%)\n")
print((out * 100).round(2))

from AlgorithmImports import *
from typing import Dict, List
import numpy as np


class ZephyrMacroAllocator(QCAlgorithm):
    """
    Multi-asset, regime-aware allocation strategy using momentum,
    trend filtering, and volatility-adjusted win-rate confidence.

    Volatility is used ONLY to penalize noisy signals in edge construction.
    There is NO volatility targeting or exposure scaling.
    """

    # ====================================================
    # initialize
    # ====================================================
    def initialize(self) -> None:
        """
        Initialize the Zephyr Macro Allocator algorithm.

        Sets global algorithm parameters, user-configurable options, asset
        universes, indicators, warm-up period, and the scheduled rebalance event.

        The strategy constructs a multi-asset portfolio using:
        - Multi-horizon momentum
        - SMA-based trend filtering
        - Win-rate and volatility-adjusted confidence scoring
        - Optional sector and treasury regime logic

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method configures the algorithm state and does not return a value.
        """
        self.set_start_date(2012, 1, 1)
        self.set_cash(100_000)

        # ============================
        # user options
        # ============================
        self.enable_sma_filter = True
        self.enable_treasury_kill_switch = True
        self.use_group_momentum = False
        self.winrate_lookback = 126
        self.vol_lookback = 126

        self.sma_period = 147
        self.bond_sma_period = 126
        self.crypto_cap = 0.10

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # ============================
        # asset groups
        # ============================
        self.group_tickers = {
            "real": ["GLD", "DBC"],
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
            "high_yield_bonds": ["SHYG", "HYG"],
            "equities": ["VTI", "VEA", "VWO"],
            "equity_income": ["VIG", "VYM", "VIGI", "VYMI"],
            "crypto": ["BTCUSD", "ETHUSD"],
            "cash": ["SHV"]
        }

        self.symbols: Dict[str, List[Symbol]] = {}

        for group, tickers in self.group_tickers.items():
            self.symbols[group] = []
            for ticker in tickers:
                symbol = (
                    self.add_crypto(ticker, Resolution.Daily).Symbol
                    if ticker.endswith("USD")
                    else self.add_equity(ticker, Resolution.Daily).Symbol
                )
                self.symbols[group].append(symbol)
                self.Securities[symbol].FeeModel = ConstantFeeModel(0)

        self.all_symbols = [s for v in self.symbols.values() for s in v]

        # ----------------------------
        # bond universe
        # ----------------------------
        self.bond_symbols = {
            s for g in ["corp_bonds", "treasury_bonds", "high_yield_bonds"]
            for s in self.symbols[g]
        }

        # ============================
        # indicators
        # ============================
        self.smas = {
            s: self.SMA(s, self.sma_period, Resolution.DAILY)
            for s in self.all_symbols if s not in self.bond_symbols
        }

        self.bond_smas = {
            s: self.SMA(s, self.bond_sma_period, Resolution.DAILY)
            for s in self.bond_symbols
        }

        self.set_warmup(
            max(
                self.winrate_lookback,
                self.vol_lookback,
                self.max_lookback,
                self.bond_sma_period
            )
        )

        self.Schedule.On(
            self.DateRules.MonthEnd(self.all_symbols[0]),
            self.TimeRules.BeforeMarketClose(self.all_symbols[0], 5),
            self.rebalance
        )

    # ====================================================
    # trend filter
    # ====================================================
    def passes_trend(self, symbol: Symbol) -> bool:
        """
        Determine whether an asset passes the trend filter.

        Uses a simple moving average (SMA) trend filter. Equity-like assets
        use `self.sma_period`, while bond assets use `self.bond_sma_period`.
        If the SMA filter is disabled, all assets automatically pass.

        Parameters
        ----------
        symbol : Symbol
            The QuantConnect symbol to evaluate.

        Returns
        -------
        bool
            True if the asset price is above its SMA or if trend filtering
            is disabled; False otherwise.
        """
        if not self.enable_sma_filter:
            return True

        sma = (
            self.bond_smas.get(symbol)
            if symbol in self.bond_symbols
            else self.smas.get(symbol)
        )

        return sma and sma.IsReady and self.Securities[symbol].Price > sma.Current.Value

    # ====================================================
    # NEW: treasury trend-only kill helper
    # ====================================================
    def all_treasuries_fail_trend(self) -> bool:
        """
        Check whether all treasury assets fail the trend filter.

        This helper supports the treasury-only kill switch. Risk is disabled
        only if *every* treasury bond asset is trading below its SMA.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if all treasury assets fail the trend filter; False otherwise.
            Always returns False if the SMA filter is disabled.
        """
        if not self.enable_sma_filter:
            return False

        for s in self.symbols["treasury_bonds"]:
            sma = self.bond_smas.get(s)
            if sma and sma.IsReady and self.Securities[s].Price > sma.Current.Value:
                return False 

        return True

    # ====================================================
    # rebalance
    # ====================================================
    def rebalance(self) -> None:
        """
        Perform a full portfolio rebalance using momentum and confidence scoring.

        This method liquidates existing positions, determines regime-qualified assets,
        constructs group-level edges, handles the treasury kill switch, and 
        allocates capital via SetHoldings.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Portfolio holdings are updated via SetHoldings; no value is returned.
        """
        if self.IsWarmingUp:
            return

        self.Liquidate()

        history = self.history(
            self.all_symbols,
            self.max_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        closes = history["close"].unstack(0)
        cash_symbol = self.symbols["cash"][0]

        bil_6m = self.six_month_return(cash_symbol, closes)

        corp_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["corp_bonds"]
        )
        treasury_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["treasury_bonds"]
        )
        high_yield_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["high_yield_bonds"]
        )

        risk_groups = {
            "real": self.symbols["real"],
            "corp_bonds": corp_bonds,
            "treasury_bonds": treasury_bonds,
            "high_yield_bonds": high_yield_bonds,
            "equities": self.symbols["equities"],
            "equity_income": self.symbols["equity_income"],
            "crypto": self.symbols["crypto"]
        }

        vols = {}
        edges = {}
        group_assets = {}
        group_asset_edges = {}

        # ============================
        # group + asset edge construction
        # ============================
        for group, symbols in risk_groups.items():
            eligible = []
            asset_edges = {}
            asset_adxs = {}

            for s in symbols:
                if s not in closes.columns:
                    continue
                if not self.passes_trend(s):
                    continue

                asset_momentum = self.compute_asset_momentum(s, closes)
                asset_6m = self.six_month_return(s, closes)

                if asset_6m < bil_6m or asset_momentum <= 0:
                    continue

                s_data = history.loc[s]
                current_adx = self.compute_manual_adx(
                            s_data['high'], 
                            s_data['low'], 
                            s_data['close']
                        )

                # 2. Multiply momentum by ADX to get the adjusted edge
                asset_edge = asset_momentum * current_adx
                asset_edge = asset_momentum
                eligible.append(s)
                asset_edges[s] = asset_edge

            if not eligible:
                continue

            # 1. Create a weight series from asset_edges
            edge_series = pd.Series(asset_edges)
            asset_weights = edge_series / edge_series.sum()

            # 3. Construct the Weighted Return Series
            # This is the "Weighted Group Performance" day-by-day
            group_returns_df = closes[eligible].pct_change()
            weighted_group_returns = (group_returns_df * asset_weights).sum(axis=1).dropna()

            if len(weighted_group_returns) < max(self.winrate_lookback, self.vol_lookback):
                continue

            # 4. Group ADX (Weighted Average of Asset ADXs)
            group_adx = (pd.Series(asset_adxs) * asset_weights).sum()

            # 5. Group Momentum (Weighted Average)
            group_momentum = (edge_series * asset_weights).sum()

            # 6. CALCULATE WEIGHTED WIN RATE & VOLATILITY
            # Using the log of the weighted return series
            log_group = np.log1p(weighted_group_returns)

            # Weighted Win Rate: Frequency of positive days in the weighted basket
            win_rate = float(np.mean(log_group.tail(self.winrate_lookback) > 0))

            # Weighted Group Vol: Annualized StdDev of the weighted basket
            group_vol = float(np.std(log_group.tail(self.vol_lookback)) * np.sqrt(252))

            if not np.isfinite(group_vol) or group_vol <= 0:
                continue

            vols[group] = group_vol

            # 7. Final Confidence & Edge Construction
            # Incorporating the Group ADX as a multiplier for trend quality
            confidence = (group_momentum * group_adx) / (group_vol + 1e-6)

            if self.use_group_momentum:
                edges[group] = group_momentum * group_adx
            else:
                edges[group] = win_rate * (1.0 + confidence)

            group_assets[group] = eligible
            group_asset_edges[group] = asset_edges

        if not edges:
            self.SetHoldings(cash_symbol, 1.0)
            return

        # ====================================================
        # UPDATED TREASURY KILL SWITCH (trend-only)
        # ====================================================
        if self.enable_treasury_kill_switch and self.all_treasuries_fail_trend():
            self.SetHoldings(cash_symbol, 1.0)
            self.Debug("TREASURY KILL SWITCH: all treasuries failed trend")
            return

        # ============================
        # group weights
        # ============================
        effective_edges = {g: e for g, e in edges.items()}
        total_edge = sum(effective_edges.values())

        weights = {
            g: (effective_edges[g] / total_edge)
            for g in effective_edges
        }

        # crypto cap
        if "crypto" in weights and weights["crypto"] > self.crypto_cap:
            excess = weights["crypto"] - self.crypto_cap
            weights["crypto"] = self.crypto_cap
            others = [g for g in weights if g != "crypto"]
            total_other = sum(weights[g] for g in others)
            for g in others:
                weights[g] += excess * (weights[g] / total_other)

        cash_weight = max(0.0, 1.0 - sum(weights.values()))

        # ============================
        # intra-group allocation (EDGE ONLY)
        # ============================
        for group, group_weight in weights.items():
            symbols = group_assets.get(group, [])
            if not symbols:
                continue

            edges_i = {
                s: group_asset_edges[group][s]
                for s in symbols
            }

            edge_sum = sum(edges_i.values())
            if edge_sum <= 0:
                continue

            for s, edge in edges_i.items():
                self.SetHoldings(s, group_weight * edge / edge_sum)

        self.SetHoldings(cash_symbol, cash_weight)

        self.Debug(
            "GROUPS | "
            + " | ".join(f"{g}:{w:.2f}" for g, w in sorted(weights.items()))
            + f" | cash:{cash_weight:.2f}"
        )

    # ====================================================
    # helpers
    # ====================================================
    def six_month_return(self, symbol: Symbol, closes: pd.DataFrame) -> float:
        """
        Compute the trailing six-month simple return for a single asset.

        Parameters
        ----------
        symbol : Symbol
            The asset symbol to evaluate.
        closes : pandas.DataFrame
            DataFrame of historical close prices indexed by date and symbol.

        Returns
        -------
        float
            Six-month simple return. Returns -inf if insufficient data is available.
        """
        if symbol not in closes.columns or len(closes[symbol]) < 127:
            return -np.inf
        prices = closes[symbol]
        return float(prices.iloc[-1] / prices.iloc[-127] - 1)

    def get_duration_regime_for_group(self, closes: pd.DataFrame, symbols: List[Symbol]) -> List[Symbol]:
        """
        Select bond assets based on a duration momentum regime.

        Uses short-, intermediate-, and long-duration momentum ordering
        to dynamically select which bonds are eligible for allocation.

        Parameters
        ----------
        closes : pandas.DataFrame
            DataFrame of historical close prices indexed by date and symbol.
        symbols : list[Symbol]
            Ordered list of bond symbols (short to long duration).

        Returns
        -------
        list[Symbol]
            Subset of input symbols representing the active duration regime.
        """
        def momentum(symbol):
            if symbol not in closes.columns:
                return -np.inf
            prices = closes[symbol].replace(0, np.nan).dropna()
            if len(prices) < 127:
                return -np.inf
            return np.mean([
                prices.iloc[-1] / prices.iloc[-22] - 1,
                prices.iloc[-1] / prices.iloc[-64] - 1,
                prices.iloc[-1] / prices.iloc[-127] - 1,
            ])

        valid_symbols = [s for s in symbols if s in closes.columns]
        
        if len(valid_symbols) == 3:
            short, intermediate, long = valid_symbols
            m_long = momentum(long)
            m_int = momentum(intermediate)
            m_short = momentum(short)
            
            if m_long > m_int > m_short:
                return [short, intermediate, long]
            elif m_int > m_short:
                return [short, intermediate]
            else:
                return [short]

        if len(valid_symbols) == 2:
            short, long = valid_symbols
            return [short, long] if momentum(long) > momentum(short) else [short]

        return valid_symbols

    def compute_group_momentum(self, symbols: List[Symbol], closes: pd.DataFrame) -> float:
        """
        Compute aggregate momentum for an asset group.

        Parameters
        ----------
        symbols : list[Symbol]
            Assets belonging to the group.
        closes : pandas.DataFrame
            DataFrame of historical close prices indexed by date and symbol.

        Returns
        -------
        float
            Average group momentum. Returns 0.0 if no valid assets exist.
        """
        values = []
        for symbol in symbols:
            mom = self.compute_asset_momentum(symbol, closes)

            if np.isfinite(mom):
                values.append(mom)

        return float(np.mean(values)) if values else 0.0

    def compute_asset_momentum(self, symbol: Symbol, closes: pd.DataFrame) -> float:
        """
        Compute multi-horizon momentum for a single asset.

        Parameters
        ----------
        symbol : Symbol
            Asset symbol for which momentum is computed.
        closes : pandas.DataFrame
            DataFrame of historical close prices indexed by date and symbol.

        Returns
        -------
        float
            Momentum score for the asset. Returns -inf if insufficient
            historical data is available.
        """
        if symbol not in closes.columns:
            return -np.inf
            
        prices = closes[symbol].replace(0, np.nan).dropna()
        
        if len(prices) < self.max_lookback + 1:
            return -np.inf

        return float(np.mean([
            prices.iloc[-1] / prices.iloc[-(lb + 1)] - 1
            for lb in self.momentum_lookbacks
        ]))


    def compute_manual_adx(self, high, low, close, period=14):
        # 1. Calculate +DM and -DM (Directional Movement)
        up_move = high.diff()
        down_move = -low.diff()

        # Apply your logic: If move is positive and greater than the other, keep it
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # 2. True Range (TR) - Required to normalize the DI values
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 3. Calculate the Averages (Using Wilder's Smoothing / EMA)
        # We smooth TR, +DM, and -DM over the period
        alpha = 1 / period
        smooth_plus_dm = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        smooth_minus_dm = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
        smooth_tr = tr.ewm(alpha=alpha, adjust=False).mean()

        # 4. Calculate DI+ and DI-
        plus_di = 100 * (smooth_plus_dm / smooth_tr)
        minus_di = 100 * (smooth_minus_dm / smooth_tr)

        # 5. Calculate DX (Your step: |DI+ - DI-| / |DI+ + DI-| * 100)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

        # 6. Final ADX (The average of DX over time)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx.iloc[-1] # Return the most recent value

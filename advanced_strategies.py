"""
Advanced Trading Strategies - Part 2
10 more strategies based on popular trading ideas
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class StrategyResult:
    name: str
    final_value: float
    total_invested: float
    total_return_pct: float
    num_trades: int
    winning_trades: int
    max_drawdown_pct: float
    sharpe_ratio: float
    equity_curve: pd.Series
    btc_held: float
    description: str


class AdvancedStrategies:
    """10 Advanced trading strategies."""
    
    def __init__(self, data: pd.DataFrame, initial_cash: float = 10000, commission: float = 0.001):
        self.data = data
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = []
    
    def run_all(self) -> List[StrategyResult]:
        """Run all 10 advanced strategies."""
        strategies = [
            self.strategy_1_atr_trailing_stop,
            self.strategy_2_ichimoku_cloud,
            self.strategy_3_parabolic_sar,
            self.strategy_4_stochastic,
            self.strategy_5_williams_r,
            self.strategy_6_mfi,
            self.strategy_7_donchian_channels,
            self.strategy_8_keltner_channels,
            self.strategy_9_supertrend,
            self.strategy_10_dual_thrust,
        ]
        
        for strategy_func in strategies:
            try:
                result = strategy_func()
                self.results.append(result)
                print(f"✓ {result.name}: {result.total_return_pct:+.2f}%")
            except Exception as e:
                print(f"✗ {strategy_func.__name__}: {e}")
        
        return self.results
    
    def _execute_signals(self, signals: pd.Series) -> tuple:
        """Execute trading signals."""
        cash = self.initial_cash
        btc = 0.0
        equity_curve = []
        trades = []
        entry_price = 0
        
        for date, signal in signals.items():
            price = self.data.loc[date, 'close']
            
            if signal == 1 and cash > 100:
                invest = cash * 0.99
                btc = (invest * (1 - self.commission)) / price
                cash = 0
                entry_price = price
                trades.append({'type': 'buy', 'date': date, 'price': price})
            
            elif signal == -1 and btc > 0:
                sell_value = btc * price * (1 - self.commission)
                pnl = sell_value - (btc * entry_price)
                cash = sell_value
                trades.append({'type': 'sell', 'date': date, 'price': price, 'pnl': pnl})
                btc = 0
            
            equity = cash + btc * price
            equity_curve.append(equity)
        
        if btc > 0:
            cash = btc * self.data['close'].iloc[-1] * (1 - self.commission)
        
        return pd.Series(equity_curve, index=signals.index), cash, btc, trades
    
    def _calculate_metrics(self, equity_curve: pd.Series, trades: List[dict]) -> tuple:
        """Calculate performance metrics."""
        total_trades = len([t for t in trades if t.get('type') == 'sell'])
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        returns = equity_curve.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
        
        return total_trades, winning_trades, max_dd, sharpe
    
    # ====== Strategy 1: ATR Trailing Stop ======
    def strategy_1_atr_trailing_stop(self) -> StrategyResult:
        """ATR Trailing Stop - Volatility-based trend following"""
        df = self.data.copy()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        
        # Trailing stop (3x ATR)
        multiplier = 3
        df['highest'] = df['close'].rolling(20).max()
        df['lowest'] = df['close'].rolling(20).min()
        df['long_stop'] = df['highest'] - (multiplier * atr)
        df['short_stop'] = df['lowest'] + (multiplier * atr)
        
        signals = pd.Series(0, index=df.index)
        position = 0
        
        for i in range(20, len(df)):
            price = df['close'].iloc[i]
            
            if position == 0 and price > df['highest'].iloc[i-1]:
                signals.iloc[i] = 1
                position = 1
            elif position == 1 and price < df['long_stop'].iloc[i-1]:
                signals.iloc[i] = -1
                position = 0
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="1. ATR Trailing Stop (14, 3x)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="ATR追踪止损，波动率自适应"
        )
    
    # ====== Strategy 2: Ichimoku Cloud ======
    def strategy_2_ichimoku_cloud(self) -> StrategyResult:
        """Ichimoku Cloud - Japanese trend indicator"""
        df = self.data.copy()
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        df['senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        
        signals = pd.Series(0, index=df.index)
        position = 0
        
        for i in range(52, len(df)):
            price = df['close'].iloc[i]
            tenkan = df['tenkan'].iloc[i]
            kijun = df['kijun'].iloc[i]
            
            # Buy: Price above cloud, Tenkan crosses above Kijun
            if position == 0 and tenkan > kijun and price > max(df['senkou_a'].iloc[i], df['senkou_b'].iloc[i]):
                signals.iloc[i] = 1
                position = 1
            # Sell: Price below cloud or Tenkan crosses below Kijun
            elif position == 1 and (tenkan < kijun or price < min(df['senkou_a'].iloc[i], df['senkou_b'].iloc[i])):
                signals.iloc[i] = -1
                position = 0
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="2. Ichimoku Cloud",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="一目均衡表，日本趋势指标"
        )
    
    # ====== Strategy 3: Parabolic SAR ======
    def strategy_3_parabolic_sar(self) -> StrategyResult:
        """Parabolic SAR - Stop and Reverse"""
        df = self.data.copy()
        
        af = 0.02  # Acceleration factor
        max_af = 0.2
        
        sar = [df['low'].iloc[0]]
        ep = df['high'].iloc[0]  # Extreme point
        trend = 1  # 1 = up, -1 = down
        
        for i in range(1, len(df)):
            if trend == 1:
                sar.append(sar[-1] + af * (ep - sar[-1]))
                if df['low'].iloc[i] < sar[-1]:
                    trend = -1
                    sar[-1] = ep
                    ep = df['low'].iloc[i]
                    af = 0.02
                elif df['high'].iloc[i] > ep:
                    ep = df['high'].iloc[i]
                    af = min(af + 0.02, max_af)
            else:
                sar.append(sar[-1] + af * (ep - sar[-1]))
                if df['high'].iloc[i] > sar[-1]:
                    trend = 1
                    sar[-1] = ep
                    ep = df['high'].iloc[i]
                    af = 0.02
                elif df['low'].iloc[i] < ep:
                    ep = df['low'].iloc[i]
                    af = min(af + 0.02, max_af)
        
        df['sar'] = sar
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['sar'].iloc[i] and df['close'].iloc[i-1] <= df['sar'].iloc[i-1]:
                signals.iloc[i] = 1
            elif df['close'].iloc[i] < df['sar'].iloc[i] and df['close'].iloc[i-1] >= df['sar'].iloc[i-1]:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="3. Parabolic SAR",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="抛物线转向，追踪止损反转"
        )
    
    # ====== Strategy 4: Stochastic Oscillator ======
    def strategy_4_stochastic(self) -> StrategyResult:
        """Stochastic Oscillator - Momentum indicator"""
        df = self.data.copy()
        
        period = 14
        smooth_k = 3
        smooth_d = 3
        
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        
        df['k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['k'] = df['k'].rolling(window=smooth_k).mean()
        df['d'] = df['k'].rolling(window=smooth_d).mean()
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            # Buy: %K crosses above %D from below 20
            if df['k'].iloc[i-1] < df['d'].iloc[i-1] and df['k'].iloc[i] > df['d'].iloc[i] and df['k'].iloc[i] < 20:
                signals.iloc[i] = 1
            # Sell: %K crosses below %D from above 80
            elif df['k'].iloc[i-1] > df['d'].iloc[i-1] and df['k'].iloc[i] < df['d'].iloc[i] and df['k'].iloc[i] > 80:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="4. Stochastic (14,3,3)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="随机指标，动量震荡"
        )
    
    # ====== Strategy 5: Williams %R ======
    def strategy_5_williams_r(self) -> StrategyResult:
        """Williams %R - Overbought/Oversold"""
        df = self.data.copy()
        
        period = 14
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            # Buy: Williams %R crosses above -80 from below
            if df['williams_r'].iloc[i-1] < -80 and df['williams_r'].iloc[i] >= -80:
                signals.iloc[i] = 1
            # Sell: Williams %R crosses below -20 from above
            elif df['williams_r'].iloc[i-1] > -20 and df['williams_r'].iloc[i] <= -20:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="5. Williams %R (14)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="威廉指标，超买超卖"
        )
    
    # ====== Strategy 6: MFI (Money Flow Index) ======
    def strategy_6_mfi(self) -> StrategyResult:
        """MFI - Volume-weighted RSI"""
        df = self.data.copy()
        
        period = 14
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        money_flow_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
        signed_money_flow = raw_money_flow * money_flow_sign
        
        positive_flow = signed_money_flow.where(signed_money_flow > 0, 0).rolling(window=period).sum()
        negative_flow = (-signed_money_flow.where(signed_money_flow < 0, 0)).rolling(window=period).sum()
        
        money_ratio = positive_flow / negative_flow
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['mfi'].iloc[i-1] < 20 and df['mfi'].iloc[i] >= 20:
                signals.iloc[i] = 1
            elif df['mfi'].iloc[i-1] > 80 and df['mfi'].iloc[i] <= 80:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="6. MFI (14)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="资金流量指标，成交量加权RSI"
        )
    
    # ====== Strategy 7: Donchian Channels ======
    def strategy_7_donchian_channels(self) -> StrategyResult:
        """Donchian Channels - Breakout strategy"""
        df = self.data.copy()
        period = 20
        
        df['upper'] = df['high'].rolling(window=period).max()
        df['lower'] = df['low'].rolling(window=period).min()
        df['middle'] = (df['upper'] + df['lower']) / 2
        
        signals = pd.Series(0, index=df.index)
        position = 0
        
        for i in range(period, len(df)):
            if position == 0 and df['close'].iloc[i] > df['upper'].iloc[i-1]:
                signals.iloc[i] = 1
                position = 1
            elif position == 1 and df['close'].iloc[i] < df['lower'].iloc[i-1]:
                signals.iloc[i] = -1
                position = 0
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="7. Donchian Channels (20)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="唐奇安通道，突破策略"
        )
    
    # ====== Strategy 8: Keltner Channels ======
    def strategy_8_keltner_channels(self) -> StrategyResult:
        """Keltner Channels - ATR-based bands"""
        df = self.data.copy()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(10).mean()
        
        # Middle line = EMA 20
        df['middle'] = df['close'].ewm(span=20).mean()
        df['upper'] = df['middle'] + 2 * atr
        df['lower'] = df['middle'] - 2 * atr
        
        signals = pd.Series(0, index=df.index)
        for i in range(20, len(df)):
            if df['close'].iloc[i-1] <= df['lower'].iloc[i-1] and df['close'].iloc[i] > df['lower'].iloc[i]:
                signals.iloc[i] = 1
            elif df['close'].iloc[i-1] >= df['upper'].iloc[i-1] and df['close'].iloc[i] < df['upper'].iloc[i]:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="8. Keltner Channels (20,2)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="凯尔特纳通道，ATR-based"
        )
    
    # ====== Strategy 9: Supertrend ======
    def strategy_9_supertrend(self) -> StrategyResult:
        """Supertrend - Trend following with ATR"""
        df = self.data.copy()
        
        period = 10
        multiplier = 3
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        # Basic bands
        hl2 = (df['high'] + df['low']) / 2
        df['upper'] = hl2 + multiplier * atr
        df['lower'] = hl2 - multiplier * atr
        
        # Supertrend calculation
        trend = [1] * len(df)
        supertrend = [0] * len(df)
        
        for i in range(period, len(df)):
            if df['close'].iloc[i] > df['upper'].iloc[i-1]:
                trend[i] = 1
            elif df['close'].iloc[i] < df['lower'].iloc[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
            
            if trend[i] == 1:
                supertrend[i] = df['lower'].iloc[i]
            else:
                supertrend[i] = df['upper'].iloc[i]
        
        df['trend'] = trend
        df['supertrend'] = supertrend
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['trend'].iloc[i] == 1 and df['trend'].iloc[i-1] == -1:
                signals.iloc[i] = 1
            elif df['trend'].iloc[i] == -1 and df['trend'].iloc[i-1] == 1:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="9. Supertrend (10,3)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="超级趋势，ATR趋势跟踪"
        )
    
    # ====== Strategy 10: Dual Thrust ======
    def strategy_10_dual_thrust(self) -> StrategyResult:
        """Dual Thrust - Opening range breakout"""
        df = self.data.copy()
        n = 4  # Lookback period
        
        df['range1'] = df['high'].rolling(n).max() - df['close'].rolling(n).min()
        df['range2'] = df['close'].rolling(n).max() - df['low'].rolling(n).min()
        df['range'] = df[['range1', 'range2']].max(axis=1)
        
        k1 = 0.5  # Upper multiplier
        k2 = 0.5  # Lower multiplier
        
        df['upper'] = df['open'] + k1 * df['range'].shift(1)
        df['lower'] = df['open'] - k2 * df['range'].shift(1)
        
        signals = pd.Series(0, index=df.index)
        position = 0
        
        for i in range(n+1, len(df)):
            if position == 0 and df['close'].iloc[i] > df['upper'].iloc[i]:
                signals.iloc[i] = 1
                position = 1
            elif position == 1 and df['close'].iloc[i] < df['lower'].iloc[i]:
                signals.iloc[i] = -1
                position = 0
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="10. Dual Thrust (4,0.5)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="双推力，开盘区间突破"
        )


def plot_comparison(data, results, save_path):
    """Plot comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: Price
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], label='BTC Price', color='black', linewidth=1.5)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Bitcoin Price', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Portfolio
    ax2 = axes[1]
    for i, r in enumerate(results):
        label = f"{i+1}. {r.name.split('.')[1].strip()[:18]} ({r.total_return_pct:+.1f}%)"
        ax2.plot(r.equity_curve.index, r.equity_curve, label=label, 
                color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax2.set_title('Advanced Strategies - Portfolio Value', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: Drawdown
    ax3 = axes[2]
    for i, r in enumerate(results):
        rolling_max = r.equity_curve.expanding().max()
        drawdown = (r.equity_curve - rolling_max) / rolling_max * 100
        ax3.plot(drawdown.index, drawdown, label=r.name.split('.')[1].strip()[:18], 
                color=colors[i], linewidth=1.2, alpha=0.8)
    
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Chart saved to: {save_path}")
    plt.close()


def print_summary(results):
    """Print summary."""
    print("\n" + "="*120)
    print("                                    10 ADVANCED STRATEGIES COMPARISON")
    print("="*120)
    print(f"{'Rank':>4} {'Strategy':<35} {'Return':>10} {'Max DD':>10} {'Sharpe':>8} {'Trades':>8} {'Win%':>8}")
    print("-"*120)
    
    sorted_results = sorted(results, key=lambda x: x.total_return_pct, reverse=True)
    
    for i, r in enumerate(sorted_results, 1):
        win_pct = (r.winning_trades / r.num_trades * 100) if r.num_trades > 0 else 0
        name = r.name.split('.')[1].strip() if '.' in r.name else r.name
        print(f"{i:>4} {name:<35} {r.total_return_pct:>9.2f}% {r.max_drawdown_pct:>9.2f}% "
              f"{r.sharpe_ratio:>8.2f} {r.num_trades:>8} {win_pct:>7.1f}%")
    
    print("="*120)


def main():
    print("="*70)
    print(f"{'10 ADVANCED TRADING STRATEGIES':^70}")
    print("="*70)
    
    try:
        data = pd.read_pickle('/tmp/btc_3y_data.pkl')
        period_note = "3 Years"
    except:
        data = pd.read_pickle('/tmp/btc_data_cache.pkl')
        period_note = "1 Year"
    
    print(f"\n📊 Data: {period_note}")
    print(f"   {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Price: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
    
    print(f"\n🔄 Running 10 advanced strategies...")
    print("-"*70)
    
    strategies = AdvancedStrategies(data, initial_cash=10000, commission=0.001)
    results = strategies.run_all()
    
    print_summary(results)
    
    print(f"\n📊 Generating comparison chart...")
    plot_comparison(data, results, save_path='/root/.openclaw/workspace/btc_advanced_strategies.png')
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

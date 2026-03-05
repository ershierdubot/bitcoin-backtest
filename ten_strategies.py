"""
10 Trading Strategies Implementation
实现10种常见的比特币交易策略并对比
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StrategyResult:
    """Strategy backtest result."""
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


class TradingStrategies:
    """Collection of 10 trading strategies."""
    
    def __init__(self, data: pd.DataFrame, initial_cash: float = 10000, commission: float = 0.001):
        self.data = data
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = []
    
    def run_all(self) -> List[StrategyResult]:
        """Run all 10 strategies."""
        strategies = [
            self.strategy_1_sma_crossover,
            self.strategy_2_rsi_mean_reversion,
            self.strategy_3_macd_momentum,
            self.strategy_4_bollinger_bands,
            self.strategy_5_double_ma,
            self.strategy_6_volume_weighted,
            self.strategy_7_breakout,
            self.strategy_8_grid_trading,
            self.strategy_9_dca_daily,
            self.strategy_10_buy_hold,
        ]
        
        for strategy_func in strategies:
            try:
                result = strategy_func()
                self.results.append(result)
                print(f"✓ {result.name}: {result.total_return_pct:+.2f}%")
            except Exception as e:
                print(f"✗ {strategy_func.__name__}: {e}")
        
        return self.results
    
    def _calculate_metrics(self, equity_curve: pd.Series, trades: List[dict]) -> tuple:
        """Calculate performance metrics."""
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        
        # Max drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        # Sharpe ratio
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 1 and returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
        else:
            sharpe = 0
        
        return total_trades, winning_trades, max_dd, sharpe
    
    def _execute_signals(self, signals: pd.Series) -> tuple:
        """Execute trading signals and return equity curve."""
        cash = self.initial_cash
        btc = 0.0
        equity_curve = []
        trades = []
        entry_price = 0
        
        for date, signal in signals.items():
            price = self.data.loc[date, 'close']
            
            if signal == 1 and cash > 100:  # Buy
                invest = cash * 0.99
                btc = (invest * (1 - self.commission)) / price
                cash = 0
                entry_price = price
                trades.append({'type': 'buy', 'date': date, 'price': price})
            
            elif signal == -1 and btc > 0:  # Sell
                sell_value = btc * price * (1 - self.commission)
                pnl = sell_value - (btc * entry_price)
                cash = sell_value
                trades.append({'type': 'sell', 'date': date, 'price': price, 'pnl': pnl})
                btc = 0
            
            equity = cash + btc * price
            equity_curve.append(equity)
        
        # Close at end
        if btc > 0:
            cash = btc * self.data['close'].iloc[-1] * (1 - self.commission)
        
        return pd.Series(equity_curve, index=signals.index), cash, btc, trades
    
    # ====== Strategy 1: SMA Crossover ======
    def strategy_1_sma_crossover(self) -> StrategyResult:
        """SMA 20/50 Crossover"""
        df = self.data.copy()
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[df['sma20'] > df['sma50']] = 1
        signals = signals.diff().fillna(0)
        signals[signals > 0] = 1
        signals[signals < 0] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="1. SMA Crossover (20/50)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="经典双均线交叉策略"
        )
    
    # ====== Strategy 2: RSI Mean Reversion ======
    def strategy_2_rsi_mean_reversion(self) -> StrategyResult:
        """RSI Oversold/Overbought"""
        df = self.data.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['rsi'].iloc[i-1] < 30 and df['rsi'].iloc[i] >= 30:
                signals.iloc[i] = 1
            elif df['rsi'].iloc[i-1] > 70 and df['rsi'].iloc[i] <= 70:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="2. RSI Mean Reversion (30/70)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="RSI超卖买入、超买卖出"
        )
    
    # ====== Strategy 3: MACD Momentum ======
    def strategy_3_macd_momentum(self) -> StrategyResult:
        """MACD Signal Line Crossover"""
        df = self.data.copy()
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['signal'] = df['macd'].ewm(span=9).mean()
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['macd'].iloc[i-1] < df['signal'].iloc[i-1] and df['macd'].iloc[i] > df['signal'].iloc[i]:
                signals.iloc[i] = 1
            elif df['macd'].iloc[i-1] > df['signal'].iloc[i-1] and df['macd'].iloc[i] < df['signal'].iloc[i]:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="3. MACD Momentum (12/26/9)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="MACD动量指标交叉"
        )
    
    # ====== Strategy 4: Bollinger Bands ======
    def strategy_4_bollinger_bands(self) -> StrategyResult:
        """Bollinger Bands Mean Reversion"""
        df = self.data.copy()
        df['sma20'] = df['close'].rolling(20).mean()
        df['std20'] = df['close'].rolling(20).std()
        df['upper'] = df['sma20'] + 2 * df['std20']
        df['lower'] = df['sma20'] - 2 * df['std20']
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] < df['lower'].iloc[i-1] and df['close'].iloc[i] >= df['lower'].iloc[i]:
                signals.iloc[i] = 1
            elif df['close'].iloc[i-1] > df['upper'].iloc[i-1] and df['close'].iloc[i] <= df['upper'].iloc[i]:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="4. Bollinger Bands (20, 2σ)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="布林带均值回归"
        )
    
    # ====== Strategy 5: Double MA (EMA) ======
    def strategy_5_double_ma(self) -> StrategyResult:
        """EMA 12/26 Crossover"""
        df = self.data.copy()
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[df['ema12'] > df['ema26']] = 1
        signals = signals.diff().fillna(0)
        signals[signals > 0] = 1
        signals[signals < 0] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="5. Double EMA (12/26)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="双指数均线交叉"
        )
    
    # ====== Strategy 6: Volume Weighted ======
    def strategy_6_volume_weighted(self) -> StrategyResult:
        """Volume Weighted Average Price (VWAP) Strategy"""
        df = self.data.copy()
        # Use price*volume as proxy for volume
        df['tpv'] = df['close'] * df['volume']
        df['vwap'] = df['tpv'].rolling(20).sum() / df['volume'].rolling(20).sum()
        
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['vwap'].iloc[i] and df['close'].iloc[i-1] <= df['vwap'].iloc[i-1]:
                signals.iloc[i] = 1
            elif df['close'].iloc[i] < df['vwap'].iloc[i] and df['close'].iloc[i-1] >= df['vwap'].iloc[i-1]:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="6. VWAP Break (20)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="成交量加权均价突破"
        )
    
    # ====== Strategy 7: Price Breakout ======
    def strategy_7_breakout(self) -> StrategyResult:
        """20-Day High/Low Breakout"""
        df = self.data.copy()
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        signals = pd.Series(0, index=df.index)
        for i in range(20, len(df)):
            if df['close'].iloc[i] > df['high_20'].iloc[i-1]:
                signals.iloc[i] = 1
            elif df['close'].iloc[i] < df['low_20'].iloc[i-1]:
                signals.iloc[i] = -1
        
        equity, final_value, btc, trades = self._execute_signals(signals)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="7. Price Breakout (20-day)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="20日高低点突破"
        )
    
    # ====== Strategy 8: Grid Trading ======
    def strategy_8_grid_trading(self) -> StrategyResult:
        """Simple Grid Trading (5% intervals)"""
        cash = self.initial_cash
        btc = 0.0
        equity_curve = []
        trades = []
        
        # Set grid levels
        start_price = self.data['close'].iloc[0]
        grid_levels = [start_price * (0.85 + i * 0.05) for i in range(7)]  # 85% to 115%
        last_trade_price = start_price
        
        for date, row in self.data.iterrows():
            price = row['close']
            
            # Buy when price drops to a lower grid level
            for level in sorted(grid_levels, reverse=True):
                if price <= level and last_trade_price > level and cash > 100:
                    buy_amount = cash / 5  # Use 20% of cash each time
                    btc += (buy_amount * (1 - self.commission)) / price
                    cash -= buy_amount
                    trades.append({'type': 'buy', 'price': price, 'pnl': 0})
                    last_trade_price = price
                    break
            
            # Sell when price rises to a higher grid level
            for level in sorted(grid_levels):
                if price >= level and last_trade_price < level and btc > 0.0001:
                    sell_value = btc * price * (1 - self.commission)
                    cost = btc * last_trade_price if last_trade_price > 0 else 0
                    pnl = sell_value - cost
                    trades.append({'type': 'sell', 'price': price, 'pnl': pnl})
                    cash += sell_value
                    btc = 0
                    last_trade_price = price
                    break
            
            equity = cash + btc * price
            equity_curve.append(equity)
        
        final_value = cash + btc * self.data['close'].iloc[-1] * (1 - self.commission)
        equity = pd.Series(equity_curve, index=self.data.index)
        total_trades, winning_trades, max_dd, sharpe = self._calculate_metrics(equity, trades)
        
        return StrategyResult(
            name="8. Grid Trading (5%)",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=total_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="网格交易(5%间隔)"
        )
    
    # ====== Strategy 9: DCA (Dollar Cost Averaging) ======
    def strategy_9_dca_daily(self) -> StrategyResult:
        """Daily DCA Strategy"""
        daily_amount = 100
        cash = self.initial_cash
        btc = 0.0
        equity_curve = []
        total_invested = 0
        
        for i, (date, row) in enumerate(self.data.iterrows()):
            price = row['close']
            
            # Buy daily with fixed amount
            if cash >= daily_amount:
                buy_amount = min(daily_amount, cash)
                btc += (buy_amount * (1 - self.commission)) / price
                cash -= buy_amount
                total_invested += buy_amount
            
            equity = cash + btc * price
            equity_curve.append(equity)
        
        final_value = cash + btc * self.data['close'].iloc[-1] * (1 - self.commission)
        equity = pd.Series(equity_curve, index=self.data.index)
        
        # Calculate metrics
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        returns = equity.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
        
        return StrategyResult(
            name="9. Daily DCA ($100/day)",
            final_value=final_value,
            total_invested=total_invested,
            total_return_pct=(final_value - total_invested) / total_invested * 100,
            num_trades=len(self.data),
            winning_trades=0,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="每日定投100美元"
        )
    
    # ====== Strategy 10: Buy and Hold ======
    def strategy_10_buy_hold(self) -> StrategyResult:
        """Simple Buy and Hold"""
        price_start = self.data['close'].iloc[0]
        price_end = self.data['close'].iloc[-1]
        
        btc = (self.initial_cash * (1 - self.commission)) / price_start
        final_value = btc * price_end * (1 - self.commission)
        
        equity_curve = [self.initial_cash] + [btc * p for p in self.data['close'].iloc[1:]]
        equity = pd.Series(equity_curve, index=self.data.index)
        
        # Calculate metrics
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        returns = equity.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
        
        return StrategyResult(
            name="10. Buy & Hold",
            final_value=final_value,
            total_invested=self.initial_cash,
            total_return_pct=(final_value - self.initial_cash) / self.initial_cash * 100,
            num_trades=1,
            winning_trades=1 if final_value > self.initial_cash else 0,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity,
            btc_held=btc,
            description="买入并持有"
        )


def plot_comparison(data, results, save_path):
    """Plot all strategies comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: Bitcoin Price
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], label='BTC Price', color='black', linewidth=1.5)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Bitcoin Price', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Portfolio Value
    ax2 = axes[1]
    for i, r in enumerate(results):
        label = f"{r.name.split('.')[0]}. {r.name.split('.')[1].strip()[:20]} ({r.total_return_pct:+.1f}%)"
        ax2.plot(r.equity_curve.index, r.equity_curve, label=label, 
                color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial $10k')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax2.set_title('Strategy Comparison - Portfolio Value', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: Drawdown
    ax3 = axes[2]
    for i, r in enumerate(results):
        rolling_max = r.equity_curve.expanding().max()
        drawdown = (r.equity_curve - rolling_max) / rolling_max * 100
        ax3.plot(drawdown.index, drawdown, label=r.name.split('.')[1].strip()[:20], 
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
    """Print results summary table."""
    print("\n" + "="*120)
    print("                                            10 STRATEGIES COMPARISON RESULTS")
    print("="*120)
    print(f"{'Rank':>4} {'Strategy':<30} {'Return':>10} {'Max DD':>10} {'Sharpe':>8} {'Trades':>8} {'Win%':>8} {'BTC Held':>12}")
    print("-"*120)
    
    # Sort by return
    sorted_results = sorted(results, key=lambda x: x.total_return_pct, reverse=True)
    
    for i, r in enumerate(sorted_results, 1):
        win_pct = (r.winning_trades / r.num_trades * 100) if r.num_trades > 0 else 0
        name = r.name.split('.')[1].strip() if '.' in r.name else r.name
        print(f"{i:>4} {name:<30} {r.total_return_pct:>9.2f}% {r.max_drawdown_pct:>9.2f}% "
              f"{r.sharpe_ratio:>8.2f} {r.num_trades:>8} {win_pct:>7.1f}% {r.btc_held:>11.6f}")
    
    print("="*120)


def main():
    print("="*70)
    print(f"{'10 TRADING STRATEGIES COMPARISON':^70}")
    print("="*70)
    
    # Try to load 3-year data, fallback to 1-year
    try:
        data = pd.read_pickle('/tmp/btc_3y_data.pkl')
        period_note = "3 Years"
    except:
        data = pd.read_pickle('/tmp/btc_data_cache.pkl')
        period_note = "1 Year (3-year data unavailable - API limits)"
    
    print(f"\n📊 Data Period: {period_note}")
    print(f"   {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Price: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
    print(f"   Days: {len(data)}")
    
    print(f"\n🔄 Running 10 strategies with $10,000 initial capital...")
    print("-"*70)
    
    strategies = TradingStrategies(data, initial_cash=10000, commission=0.001)
    results = strategies.run_all()
    
    print_summary(results)
    
    print(f"\n📊 Generating comparison chart...")
    plot_comparison(data, results, save_path='/root/.openclaw/workspace/btc_10strategies_comparison.png')
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

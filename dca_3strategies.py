"""
DCA Strategy Comparison - 3 Variants
1. DCA + Take Profit (10%)
2. DCA + Take Profit (10%) + Stop Loss (10%)
3. Pure DCA (Buy & Hold, never sell)
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
class DCAResult:
    """Results of DCA backtest."""
    strategy_name: str
    daily_amount: float
    final_value: float
    total_invested: float
    total_btc: float
    total_profit: float
    total_return_pct: float
    num_trades: int
    winning_trades: int
    max_drawdown_pct: float
    equity_curve: pd.Series
    btc_curve: pd.Series
    invested_curve: pd.Series


class DCABacktest:
    """
    DCA strategies backtest engine
    """
    
    def __init__(
        self,
        daily_amount: float = 100.0,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        commission: float = 0.001,
    ):
        self.daily_amount = daily_amount
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.commission = commission
    
    def run(self, data: pd.DataFrame) -> DCAResult:
        """Run DCA backtest."""
        cash_balance = 0.0
        btc_holding = 0.0
        total_invested = 0.0
        total_cost_basis = 0.0
        
        num_trades = 0
        winning_trades = 0
        
        equity_curve = []
        btc_curve = []
        invested_curve = []
        
        for date, row in data.iterrows():
            price = row['close']
            
            # 每日定投
            invest_amount = self.daily_amount
            if cash_balance > 0:
                invest_amount += cash_balance
                cash_balance = 0
            
            actual_invest = invest_amount * (1 - self.commission)
            btc_bought = actual_invest / price
            
            btc_holding += btc_bought
            total_invested += invest_amount
            total_cost_basis += actual_invest
            
            current_value = btc_holding * price
            
            # 检查止盈止损条件（如果有设置）
            if total_cost_basis > 0 and (self.take_profit_pct or self.stop_loss_pct):
                profit_pct = (current_value - total_cost_basis) / total_cost_basis
                
                should_sell = False
                is_win = False
                
                # 止盈
                if self.take_profit_pct and profit_pct >= self.take_profit_pct:
                    should_sell = True
                    is_win = True
                
                # 止损
                if self.stop_loss_pct and profit_pct <= -self.stop_loss_pct:
                    should_sell = True
                    is_win = False
                
                if should_sell and btc_holding > 0:
                    # 清仓
                    exit_value = current_value * (1 - self.commission)
                    
                    if is_win:
                        winning_trades += 1
                    
                    cash_balance = exit_value
                    btc_holding = 0.0
                    total_cost_basis = 0.0
                    num_trades += 1
            
            # 记录曲线
            total_value = cash_balance + btc_holding * price
            equity_curve.append(total_value)
            btc_curve.append(btc_holding)
            invested_curve.append(total_invested)
        
        # 最终价值
        final_price = data['close'].iloc[-1]
        final_value = cash_balance + btc_holding * final_price
        
        total_profit = final_value - total_invested
        total_return_pct = (total_profit / total_invested * 100) if total_invested > 0 else 0
        
        # 最大回撤
        equity_series = pd.Series(equity_curve, index=data.index)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100
        
        # 纯定投情况下的最后btc持仓
        final_btc = btc_holding
        
        return DCAResult(
            strategy_name=self._get_strategy_name(),
            daily_amount=self.daily_amount,
            final_value=final_value,
            total_invested=total_invested,
            total_btc=final_btc,
            total_profit=total_profit,
            total_return_pct=total_return_pct,
            num_trades=num_trades,
            winning_trades=winning_trades,
            max_drawdown_pct=max_drawdown_pct,
            equity_curve=equity_series,
            btc_curve=pd.Series(btc_curve, index=data.index),
            invested_curve=pd.Series(invested_curve, index=data.index)
        )
    
    def _get_strategy_name(self):
        """Generate strategy name."""
        if self.take_profit_pct and self.stop_loss_pct:
            return f"DCA + TP{int(self.take_profit_pct*100)}% + SL{int(self.stop_loss_pct*100)}%"
        elif self.take_profit_pct:
            return f"DCA + TP{int(self.take_profit_pct*100)}%"
        else:
            return "Pure DCA (Hold)"


def print_comparison(results: List[DCAResult]):
    """Print comparison table."""
    print("\n" + "="*100)
    print("                                   DCA STRATEGY COMPARISON")
    print("="*100)
    print(f"{'Strategy':<35} {'Return':>12} {'Max DD':>10} {'Trades':>8} {'Win Rate':>10} {'Final BTC':>12}")
    print("-"*100)
    
    for r in results:
        win_rate = (r.winning_trades / r.num_trades * 100) if r.num_trades > 0 else 0
        print(f"{r.strategy_name:<35} "
              f"{r.total_return_pct:>11.2f}% "
              f"{r.max_drawdown_pct:>9.2f}% "
              f"{r.num_trades:>8} "
              f"{win_rate:>9.1f}% "
              f"{r.total_btc:>11.6f}")
    
    print("="*100)


def plot_dca_comparison(data, results, save_path=None):
    """Plot comparison of all DCA strategies."""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.2, 1, 1])
    
    colors = {
        'DCA + TP10%': '#2E86AB',
        'DCA + TP10% + SL10%': '#A23B72',
        'Pure DCA (Hold)': '#F18F01'
    }
    
    # Plot 1: Bitcoin Price
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['close'], label='BTC Price', color='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Bitcoin Price', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Portfolio Value (main comparison)
    ax2 = fig.add_subplot(gs[1])
    
    for r in results:
        color = colors.get(r.strategy_name, '#333333')
        label = f"{r.strategy_name} ({r.total_return_pct:+.1f}%)"
        ax2.plot(r.equity_curve.index, r.equity_curve, label=label, 
                color=color, linewidth=2)
    
    # 投入基准线
    ax2.plot(results[0].invested_curve.index, results[0].invested_curve, 
            linestyle='--', alpha=0.5, color='gray', linewidth=1.5, label='Total Invested')
    
    ax2.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax2.set_title('Portfolio Value Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, ncol=1)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: BTC Holdings
    ax3 = fig.add_subplot(gs[2])
    
    for r in results:
        color = colors.get(r.strategy_name, '#333333')
        ax3.plot(r.btc_curve.index, r.btc_curve, label=r.strategy_name, 
                color=color, linewidth=1.5, alpha=0.8)
    
    ax3.set_ylabel('BTC Holdings', fontsize=11)
    ax3.set_title('Bitcoin Holdings Over Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drawdown
    ax4 = fig.add_subplot(gs[3])
    
    for r in results:
        color = colors.get(r.strategy_name, '#333333')
        rolling_max = r.equity_curve.expanding().max()
        drawdown = (r.equity_curve - rolling_max) / rolling_max * 100
        ax4.plot(drawdown.index, drawdown, label=r.strategy_name, 
                color=color, linewidth=1.5, alpha=0.8)
    
    ax4.set_ylabel('Drawdown (%)', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower left', fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Comparison plot saved to: {save_path}")
    
    plt.close()


def main():
    print("="*70)
    print(f"{'DCA STRATEGY COMPARISON - 3 VARIANTS':^70}")
    print("="*70)
    
    # Load data
    print("\n📊 Loading Bitcoin data...")
    data = pd.read_pickle('/tmp/btc_data_cache.pkl')
    print(f"✓ Loaded {len(data)} rows of data")
    print(f"  Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
    
    daily_amount = 100
    
    # 三种策略
    strategies = [
        # 1. 定投+止盈10%
        DCABacktest(daily_amount=daily_amount, take_profit_pct=0.10),
        # 2. 定投+止盈10%+止损10%
        DCABacktest(daily_amount=daily_amount, take_profit_pct=0.10, stop_loss_pct=0.10),
        # 3. 纯定投（不卖出）
        DCABacktest(daily_amount=daily_amount),
    ]
    
    results = []
    
    print(f"\n🔄 Running 3 DCA strategies (每日 ${daily_amount})...")
    for strategy in strategies:
        print(f"\n  • {strategy._get_strategy_name()}...")
        result = strategy.run(data)
        results.append(result)
        print(f"    Result: {result.total_return_pct:+.2f}% return, {result.num_trades} trades, {result.total_btc:.6f} BTC")
    
    # Print comparison
    print_comparison(results)
    
    # Plot
    print(f"\n📊 Generating comparison chart...")
    plot_dca_comparison(data, results, save_path='/root/.openclaw/workspace/btc_dca_3strategies.png')
    
    print("\n✅ All done!")


if __name__ == "__main__":
    main()

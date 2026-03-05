"""
DCA (Dollar Cost Averaging) + Take Profit Strategy
定投 + 止盈清仓策略
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DCATrade:
    """Record of a DCA cycle."""
    cycle_num: int
    start_date: datetime
    end_date: datetime = None
    total_invested: float = 0.0
    total_btc: float = 0.0
    avg_price: float = 0.0
    exit_price: float = 0.0
    exit_value: float = 0.0
    profit: float = 0.0
    profit_pct: float = 0.0
    days_held: int = 0


@dataclass
class DCAResult:
    """Results of DCA backtest."""
    strategy_name: str
    daily_amount: float
    take_profit_pct: float
    final_value: float
    total_invested: float
    total_profit: float
    total_return_pct: float
    num_cycles: int
    winning_cycles: int
    avg_cycle_days: float
    avg_cycle_return: float
    max_drawdown_pct: float
    equity_curve: pd.Series
    trades: List[DCATrade]


class DCATakeProfitBacktest:
    """
    定投 + 止盈清仓策略回测
    每天投入固定金额，当盈利达到设定比例时清仓，然后继续定投
    """
    
    def __init__(
        self,
        daily_amount: float = 100.0,      # 每日定投金额
        take_profit_pct: float = 0.10,    # 止盈比例 (10%)
        commission: float = 0.001,        # 手续费 0.1%
    ):
        self.daily_amount = daily_amount
        self.take_profit_pct = take_profit_pct
        self.commission = commission
    
    def run(self, data: pd.DataFrame) -> DCAResult:
        """
        Run DCA + Take Profit backtest.
        
        Args:
            data: DataFrame with OHLCV data (daily)
        
        Returns:
            DCAResult with performance metrics
        """
        cash_balance = 0.0          # 现金余额（清仓后的资金）
        btc_holding = 0.0           # 当前持有的BTC数量
        total_invested = 0.0        # 总投入金额
        total_cost_basis = 0.0      # 当前持仓成本
        
        trades: List[DCATrade] = []
        equity_curve = []
        
        cycle_num = 0
        cycle_start_date = None
        cycle_invested = 0.0
        cycle_btc = 0.0
        
        for date, row in data.iterrows():
            price = row['close']
            
            # 1. 每日定投
            # 使用可用资金（现金余额 + 新的定投金额）
            invest_amount = self.daily_amount
            if cash_balance > 0:
                invest_amount += cash_balance
                cash_balance = 0
            
            # 扣除手续费后的实际购买金额
            actual_invest = invest_amount * (1 - self.commission)
            btc_bought = actual_invest / price
            
            # 如果是新周期的开始
            if cycle_start_date is None:
                cycle_num += 1
                cycle_start_date = date
                cycle_invested = 0.0
                cycle_btc = 0.0
            
            btc_holding += btc_bought
            total_invested += invest_amount
            total_cost_basis += actual_invest
            cycle_invested += invest_amount
            cycle_btc += btc_bought
            
            # 当前持仓价值
            current_value = btc_holding * price
            
            # 2. 检查是否达到止盈条件
            if total_cost_basis > 0:
                profit_pct = (current_value - total_cost_basis) / total_cost_basis
                
                if profit_pct >= self.take_profit_pct:
                    # 清仓！
                    # 扣除手续费后的实际卖出金额
                    exit_value = current_value * (1 - self.commission)
                    profit = exit_value - cycle_invested
                    
                    trade = DCATrade(
                        cycle_num=cycle_num,
                        start_date=cycle_start_date,
                        end_date=date,
                        total_invested=cycle_invested,
                        total_btc=btc_holding,
                        avg_price=cycle_invested / cycle_btc if cycle_btc > 0 else 0,
                        exit_price=price,
                        exit_value=exit_value,
                        profit=profit,
                        profit_pct=profit_pct,
                        days_held=(date - cycle_start_date).days
                    )
                    trades.append(trade)
                    
                    # 重置持仓，保留现金用于下一次定投
                    cash_balance = exit_value
                    btc_holding = 0.0
                    total_cost_basis = 0.0
                    cycle_start_date = None
                    cycle_invested = 0.0
                    cycle_btc = 0.0
            
            # 记录权益曲线
            total_value = cash_balance + btc_holding * price
            equity_curve.append(total_value)
        
        # 计算最终价值（包含未清仓的持仓）
        final_price = data['close'].iloc[-1]
        final_btc_value = btc_holding * final_price
        final_value = cash_balance + final_btc_value
        
        # 计算统计指标
        total_profit = final_value - total_invested
        total_return_pct = (total_profit / total_invested * 100) if total_invested > 0 else 0
        
        winning_cycles = sum(1 for t in trades if t.profit > 0)
        avg_cycle_days = np.mean([t.days_held for t in trades]) if trades else 0
        avg_cycle_return = np.mean([t.profit_pct for t in trades]) if trades else 0
        
        # 最大回撤
        equity_series = pd.Series(equity_curve, index=data.index)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100
        
        return DCAResult(
            strategy_name=f"DCA_{self.daily_amount}_TP{int(self.take_profit_pct*100)}pct",
            daily_amount=self.daily_amount,
            take_profit_pct=self.take_profit_pct,
            final_value=final_value,
            total_invested=total_invested,
            total_profit=total_profit,
            total_return_pct=total_return_pct,
            num_cycles=len(trades),
            winning_cycles=winning_cycles,
            avg_cycle_days=avg_cycle_days,
            avg_cycle_return=avg_cycle_return,
            max_drawdown_pct=max_drawdown_pct,
            equity_curve=equity_series,
            trades=trades
        )


def print_dca_results(result: DCAResult):
    """Print DCA backtest results."""
    print("\n" + "="*70)
    print(f"{'DCA + TAKE PROFIT BACKTEST RESULTS':^70}")
    print("="*70)
    print(f"Strategy:           每日定投 ${result.daily_amount:.2f}, 止盈 {result.take_profit_pct*100:.0f}%")
    print(f"Period:             {result.equity_curve.index[0].strftime('%Y-%m-%d')} to {result.equity_curve.index[-1].strftime('%Y-%m-%d')}")
    print("-"*70)
    print(f"总投入金额:         ${result.total_invested:>15,.2f}")
    print(f"最终价值:           ${result.final_value:>15,.2f}")
    print(f"总盈亏:             ${result.total_profit:>15,.2f} ({result.total_return_pct:+.2f}%)")
    print("-"*70)
    print(f"完整定投周期数:     {result.num_cycles:>15}")
    print(f"成功止盈次数:       {result.winning_cycles:>15}")
    print(f"平均周期天数:       {result.avg_cycle_days:>15.1f} 天")
    print(f"平均周期收益:       {result.avg_cycle_return*100:>15.2f}%")
    print(f"最大回撤:           {result.max_drawdown_pct:>15.2f}%")
    print("="*70)
    
    if result.trades:
        print("\n详细交易记录:")
        print("-"*70)
        print(f"{'Cycle':>6} {'Start Date':>12} {'End Date':>12} {'Days':>6} {'Invested':>12} {'Profit':>10} {'Profit%':>8}")
        print("-"*70)
        for t in result.trades:
            print(f"{t.cycle_num:>6} {t.start_date.strftime('%Y-%m-%d'):>12} {t.end_date.strftime('%Y-%m-%d'):>12} "
                  f"{t.days_held:>6} ${t.total_invested:>10,.2f} ${t.profit:>8,.2f} {t.profit_pct*100:>7.1f}%")


def plot_dca_comparison(data, results, save_path=None):
    """Plot comparison of DCA strategies."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    # Plot 1: Bitcoin Price
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], label='BTC Price', color='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Bitcoin Price', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Portfolio Value
    ax2 = axes[1]
    for i, result in enumerate(results):
        name = f"{result.daily_amount:.0f}/day, TP{result.take_profit_pct*100:.0f}% ({result.total_return_pct:+.1f}%)"
        ax2.plot(result.equity_curve.index, result.equity_curve, 
                label=name, color=colors[i % len(colors)], linewidth=1.5)
    
    # 添加投入基准线（如果不止盈，纯定投的价值）
    for result in results:
        invested_curve = pd.Series(
            [result.daily_amount * (i+1) for i in range(len(result.equity_curve))],
            index=result.equity_curve.index
        )
        ax2.plot(invested_curve.index, invested_curve, 
                linestyle='--', alpha=0.3, color='gray')
        break  # 只画一条基准线
    
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax2.set_title('DCA Strategy Comparison - Portfolio Value', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, ncol=1)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: Drawdown
    ax3 = axes[2]
    for i, result in enumerate(results):
        rolling_max = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
        name = f"{result.daily_amount:.0f}/day, TP{result.take_profit_pct*100:.0f}%"
        ax3.plot(drawdown.index, drawdown, label=name, color=colors[i % len(colors)], linewidth=1.2, alpha=0.8)
    
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=10, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {save_path}")
    
    plt.close()


def main():
    print("="*60)
    print(f"{'DCA + TAKE PROFIT STRATEGY TEST':^60}")
    print("="*60)
    
    # Load cached data
    print("\n📊 Loading Bitcoin data...")
    data = pd.read_pickle('/tmp/btc_data_cache.pkl')
    print(f"✓ Loaded {len(data)} rows of data")
    print(f"  Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
    
    # Test different DCA parameters
    configs = [
        (100, 0.10),   # 每日100，止盈10%
        (100, 0.15),   # 每日100，止盈15%
        (50, 0.10),    # 每日50，止盈10%
    ]
    
    results = []
    
    print(f"\n🔄 Testing {len(configs)} DCA configurations...")
    for daily_amount, take_profit_pct in configs:
        print(f"\n  • Testing: 每日 ${daily_amount}, 止盈 {take_profit_pct*100:.0f}%...")
        
        backtest = DCATakeProfitBacktest(
            daily_amount=daily_amount,
            take_profit_pct=take_profit_pct,
            commission=0.001
        )
        result = backtest.run(data)
        results.append(result)
        
        print(f"    Result: 总收益 {result.total_return_pct:+.2f}%, {result.num_cycles} 个完整周期")
    
    # Print detailed results for first config
    print("\n" + "="*70)
    print("详细结果 - 每日 $100, 止盈 10%:")
    print("="*70)
    print_dca_results(results[0])
    
    # Plot comparison
    print(f"\n📊 Generating comparison chart...")
    plot_dca_comparison(data, results, save_path='/root/.openclaw/workspace/btc_dca_comparison.png')
    
    print("\n✅ All done!")


if __name__ == "__main__":
    main()

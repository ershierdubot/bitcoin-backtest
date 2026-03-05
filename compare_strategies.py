"""
Compare all strategies in one plot.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import requests
from datetime import datetime, timedelta
from strategy import get_strategy
from backtest import BacktestEngine


def fetch_btc_data_cached(cache_path='/tmp/btc_data_cache.pkl'):
    """Load cached Bitcoin data."""
    df = pd.read_pickle(cache_path)
    return df


def run_strategy(name, data, initial_cash=10000, commission=0.001, position_size=1.0):
    """Run a single strategy and return results."""
    strategy = get_strategy(name)
    signals = strategy.generate_signals(data)
    
    engine = BacktestEngine(
        initial_cash=initial_cash,
        commission=commission
    )
    result = engine.run(data, signals, position_size=position_size)
    result.strategy_name = strategy.get_name()
    
    return result


def plot_comparison(data, results, save_path=None):
    """Plot comparison of all strategies with proper date formatting."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Colors for different strategies
    colors = {
        'SMA_Crossover_20_50': '#2E86AB',
        'RSI_14_30_70': '#A23B72',
        'MACD_12_26_9': '#F18F01',
        'Buy_And_Hold': '#C73E1D'
    }
    
    # Plot 1: Bitcoin Price
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], label='BTC Price', color='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Bitcoin Price', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Portfolio Value (Equity Curves)
    ax2 = axes[1]
    ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Initial Cash')
    
    for result in results:
        name = result.strategy_name
        color = colors.get(name, '#333333')
        total_return = result.total_return_pct
        ax2.plot(result.equity_curve.index, result.equity_curve, 
                label=f'{name} ({total_return:+.1f}%)', 
                color=color, linewidth=1.5)
    
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax2.set_title('Strategy Comparison - Equity Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: Drawdown
    ax3 = axes[2]
    for result in results:
        name = result.strategy_name
        color = colors.get(name, '#333333')
        rolling_max = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
        ax3.plot(drawdown.index, drawdown, label=name, color=color, linewidth=1.2, alpha=0.8)
    
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=10, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Format x-axis with dates - show every 2 months
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Comparison plot saved to: {save_path}")
    
    plt.close()


def print_summary(results):
    """Print summary table of all strategies."""
    print("\n" + "="*90)
    print("                          STRATEGY COMPARISON SUMMARY")
    print("="*90)
    print(f"{'Strategy':<25} {'Return':>12} {'Max DD':>12} {'Sharpe':>10} {'Trades':>8} {'Win Rate':>10}")
    print("-"*90)
    
    for result in results:
        print(f"{result.strategy_name:<25} "
              f"{result.total_return_pct:>11.2f}% "
              f"{result.max_drawdown:>11.2f}% "
              f"{result.sharpe_ratio:>10.2f} "
              f"{result.total_trades:>8} "
              f"{result.win_rate:>9.1f}%")
    
    print("="*90)


def main():
    print("="*60)
    print(f"{'BITCOIN STRATEGY COMPARISON':^60}")
    print("="*60)
    
    # Load cached data
    print(f"\n📊 Loading Bitcoin data...")
    data = fetch_btc_data_cached()
    print(f"✓ Fetched {len(data)} rows of data")
    print(f"  Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
    
    # Run all strategies
    strategies = ['sma', 'rsi', 'macd', 'buy_hold']
    results = []
    
    print(f"\n🔄 Testing {len(strategies)} strategies...")
    for strategy_name in strategies:
        print(f"  • Running {strategy_name.upper()}...", end=' ')
        result = run_strategy(strategy_name, data)
        results.append(result)
        print(f"Done ({result.total_return_pct:+.2f}%)")
    
    # Print summary
    print_summary(results)
    
    # Plot comparison
    print(f"\n📊 Generating comparison chart...")
    plot_comparison(data, results, save_path='/root/.openclaw/workspace/btc_strategy_comparison.png')
    
    print("\n✅ All done!")


if __name__ == "__main__":
    main()

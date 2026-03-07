"""
Main entry point for trading backtest system.
Supports Bitcoin (BTC-USD) and CSI 300 (000300.SS)
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from data_fetcher import fetch_data_by_symbol, SUPPORTED_SYMBOLS
from strategy import get_strategy
from backtest import BacktestEngine, print_backtest_results


def plot_results(data, signals, result, save_path=None, currency_suffix='$'):
    """Plot the backtest results."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Get asset name from result
    asset_name = result.strategy_name.split('_')[0] if hasattr(result, 'asset_name') else 'Asset'
    
    # Plot 1: Price and trades
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], label='Price', color='black', linewidth=1)
    
    # Mark buy and sell signals
    buy_signals = signals[signals == 1]
    sell_signals = signals[signals == -1]
    
    if len(buy_signals) > 0:
        ax1.scatter(
            buy_signals.index,
            data.loc[buy_signals.index, 'close'],
            marker='^',
            color='green',
            s=150,
            label='Buy',
            zorder=5,
            edgecolors='darkgreen',
            linewidths=1
        )
    
    if len(sell_signals) > 0:
        ax1.scatter(
            sell_signals.index,
            data.loc[sell_signals.index, 'close'],
            marker='v',
            color='red',
            s=150,
            label='Sell',
            zorder=5,
            edgecolors='darkred',
            linewidths=1
        )
    
    ax1.set_ylabel(f'Price ({currency_suffix})', fontsize=12)
    title_asset = "Bitcoin" if currency_suffix == '$' else "CSI 300"
    ax1.set_title(f'{title_asset} Backtest - {result.strategy_name}\nReturn: {result.total_return_pct:+.2f}% | Win Rate: {result.win_rate:.1f}% | Trades: {result.total_trades}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity curve
    ax2 = axes[1]
    ax2.plot(result.equity_curve.index, result.equity_curve, label='Portfolio Value', color='blue', linewidth=1.5)
    ax2.axhline(y=result.initial_cash, color='gray', linestyle='--', alpha=0.5, label='Initial Cash')
    ax2.set_ylabel(f'Portfolio Value ({currency_suffix})', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    ax3 = axes[2]
    rolling_max = result.equity_curve.expanding().max()
    drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
    ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax3.plot(drawdown.index, drawdown, color='red', linewidth=1)
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Drawdown', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.close()  # Close the figure instead of show()


def main():
    parser = argparse.ArgumentParser(description='Trading Backtest System - Bitcoin & CSI 300')
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC-USD',
        choices=['BTC-USD', '000300.SS'],
        help='Trading symbol (default: BTC-USD, use 000300.SS for CSI 300)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='sma',
        choices=['sma', 'rsi', 'macd', 'buy_hold'],
        help='Trading strategy to use (default: sma)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='1y',
        help='Data period (default: 1y)'
    )
    parser.add_argument(
        '--initial-cash',
        type=float,
        default=10000.0,
        help='Initial cash amount (default: 10000)'
    )
    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (default: 0.001 = 0.1%%)'
    )
    parser.add_argument(
        '--position-size',
        type=float,
        default=1.0,
        help='Position size as fraction of portfolio (default: 1.0)'
    )
    parser.add_argument(
        '--short-window',
        type=int,
        default=20,
        help='Short MA window for SMA strategy (default: 20)'
    )
    parser.add_argument(
        '--long-window',
        type=int,
        default=50,
        help='Long MA window for SMA strategy (default: 50)'
    )
    parser.add_argument(
        '--rsi-period',
        type=int,
        default=14,
        help='RSI period (default: 14)'
    )
    parser.add_argument(
        '--rsi-oversold',
        type=int,
        default=30,
        help='RSI oversold threshold (default: 30)'
    )
    parser.add_argument(
        '--rsi-overbought',
        type=int,
        default=70,
        help='RSI overbought threshold (default: 70)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show plot after backtest'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        help='Save plot to file'
    )
    
    args = parser.parse_args()
    
    # Get symbol info
    symbol_info = SUPPORTED_SYMBOLS.get(args.symbol, {"name": args.symbol, "currency": "USD", "suffix": "$"})
    currency_suffix = symbol_info["suffix"]
    asset_name = symbol_info["name"]
    
    print("="*60)
    print(f"{asset_name.upper()} BACKTEST SYSTEM".center(60))
    print("="*60)
    
    # Fetch data
    print(f"\n📊 Fetching {asset_name} data...")
    data = fetch_data_by_symbol(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        period=args.period
    )
    print(f"✓ Fetched {len(data)} rows of data")
    print(f"  Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: {currency_suffix}{data['close'].min():,.2f} - {currency_suffix}{data['close'].max():,.2f}")
    
    # Get strategy
    print(f"\n📈 Initializing strategy: {args.strategy.upper()}")
    
    strategy_kwargs = {}
    if args.strategy == 'sma':
        strategy_kwargs = {'short_window': args.short_window, 'long_window': args.long_window}
    elif args.strategy == 'rsi':
        strategy_kwargs = {
            'period': args.rsi_period,
            'oversold': args.rsi_oversold,
            'overbought': args.rsi_overbought
        }
    
    strategy = get_strategy(args.strategy, **strategy_kwargs)
    print(f"✓ Strategy: {strategy.get_name()}")
    
    # Generate signals
    print(f"\n🔄 Generating trading signals...")
    signals = strategy.generate_signals(data)
    num_signals = (signals != 0).sum()
    print(f"✓ Generated {num_signals} trading signals")
    
    # Run backtest
    print(f"\n💰 Running backtest...")
    print(f"  Initial cash: {currency_suffix}{args.initial_cash:,.2f}")
    print(f"  Commission: {args.commission*100:.2f}%")
    print(f"  Position size: {args.position_size*100:.0f}%")
    
    engine = BacktestEngine(
        initial_cash=args.initial_cash,
        commission=args.commission
    )
    result = engine.run(data, signals, position_size=args.position_size)
    result.strategy_name = strategy.get_name()
    
    # Print results
    print_backtest_results(result, currency_suffix=currency_suffix)
    
    # Plot if requested
    if args.plot or args.save_plot:
        print("\n📊 Generating plot...")
        plot_results(data, signals, result, save_path=args.save_plot, currency_suffix=currency_suffix)
    
    print("\n✅ Backtest complete!")
    
    return result


if __name__ == "__main__":
    main()

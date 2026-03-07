"""
Backtest engine for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    exit_date: datetime = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    side: str = ""  # 'long' or 'short'
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    strategy_name: str
    initial_cash: float
    final_value: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    trades: List[Trade]
    equity_curve: pd.Series
    start_date: datetime
    end_date: datetime


class BacktestEngine:
    """
    Backtest engine for trading strategies.
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.001,    # 0.1% slippage
    ):
        """
        Initialize the backtest engine.
        
        Args:
            initial_cash: Starting cash amount
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
            slippage: Slippage rate per trade
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
    
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        position_size: float = 1.0  # Use 1.0 for full position
    ) -> BacktestResult:
        """
        Run the backtest.
        
        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals (1=buy, -1=sell, 0=hold)
            position_size: Position size as fraction of portfolio (0-1)
        
        Returns:
            BacktestResult with performance metrics
        """
        cash = self.initial_cash
        position = 0.0  # BTC amount
        trades: List[Trade] = []
        current_trade: Trade = None
        equity_curve = []
        
        for i, (date, signal) in enumerate(signals.items()):
            price = data['close'].loc[date]
            portfolio_value = cash + position * price
            equity_curve.append(portfolio_value)
            
            # Buy signal
            if signal == 1 and position == 0:
                # Calculate position size
                trade_cash = cash * position_size
                # Apply slippage to entry price
                entry_price = price * (1 + self.slippage)
                position = trade_cash / entry_price * (1 - self.commission)
                cash -= trade_cash
                
                current_trade = Trade(
                    entry_date=date,
                    entry_price=entry_price,
                    size=position,
                    side='long'
                )
            
            # Sell signal
            elif signal == -1 and position > 0:
                # Apply slippage to exit price
                exit_price = price * (1 - self.slippage)
                trade_value = position * exit_price * (1 - self.commission)
                
                # Calculate P&L
                pnl = trade_value - (current_trade.size * current_trade.entry_price)
                pnl_pct = (exit_price - current_trade.entry_price) / current_trade.entry_price
                
                current_trade.exit_date = date
                current_trade.exit_price = exit_price
                current_trade.pnl = pnl
                current_trade.pnl_pct = pnl_pct
                trades.append(current_trade)
                
                cash += trade_value
                position = 0.0
                current_trade = None
        
        # Close any open position at the end
        if position > 0:
            final_price = data['close'].iloc[-1]
            trade_value = position * final_price * (1 - self.commission)
            
            pnl = trade_value - (current_trade.size * current_trade.entry_price)
            pnl_pct = (final_price - current_trade.entry_price) / current_trade.entry_price
            
            current_trade.exit_date = data.index[-1]
            current_trade.exit_price = final_price
            current_trade.pnl = pnl
            current_trade.pnl_pct = pnl_pct
            trades.append(current_trade)
            
            cash += trade_value
            position = 0.0
        
        final_value = cash
        equity_series = pd.Series(equity_curve, index=signals.index)
        
        return self._calculate_metrics(trades, equity_series, final_value, data)
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        final_value: float,
        data: pd.DataFrame
    ) -> BacktestResult:
        """Calculate performance metrics."""
        
        total_return = final_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100
        
        # Calculate returns for Sharpe ratio
        returns = equity_curve.pct_change().dropna()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        avg_trade_return = np.mean([t.pnl_pct for t in trades]) if trades else 0.0
        
        return BacktestResult(
            strategy_name="",
            initial_cash=self.initial_cash,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_trade_return=avg_trade_return,
            trades=trades,
            equity_curve=equity_curve,
            start_date=data.index[0],
            end_date=data.index[-1]
        )


def print_backtest_results(result: BacktestResult, currency_suffix: str = '$'):
    """Pretty print backtest results."""
    print("\n" + "="*60)
    print(f"{'BACKTEST RESULTS':^60}")
    print("="*60)
    print(f"Strategy:           {result.strategy_name}")
    print(f"Period:             {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print("-"*60)
    print(f"Initial Cash:       {currency_suffix}{result.initial_cash:>15,.2f}")
    print(f"Final Value:        {currency_suffix}{result.final_value:>15,.2f}")
    print(f"Total Return:       {currency_suffix}{result.total_return:>15,.2f} ({result.total_return_pct:+.2f}%)")
    print("-"*60)
    print(f"Sharpe Ratio:       {result.sharpe_ratio:>15.2f}")
    print(f"Max Drawdown:       {result.max_drawdown_pct:>15.2f}%")
    print("-"*60)
    print(f"Total Trades:       {result.total_trades:>15}")
    print(f"Winning Trades:     {result.winning_trades:>15}")
    print(f"Losing Trades:      {result.losing_trades:>15}")
    print(f"Win Rate:           {result.win_rate:>15.1f}%")
    print(f"Avg Trade Return:   {result.avg_trade_return*100:>15.2f}%")
    print("="*60)

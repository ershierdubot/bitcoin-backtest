"""
Trading strategies for Bitcoin backtest system.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum


class Signal(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy.
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name."""
        pass


class SmaCrossoverStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy.
    Buys when short MA crosses above long MA.
    Sells when short MA crosses below long MA.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize SMA Crossover strategy.
        
        Args:
            short_window: Short moving average window
            long_window: Long moving average window
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate SMA crossover signals."""
        signals = pd.Series(index=data.index, data=0)
        
        # Calculate moving averages
        data['sma_short'] = data['close'].rolling(window=self.short_window).mean()
        data['sma_long'] = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals[self.short_window:] = np.where(
            data['sma_short'][self.short_window:] > data['sma_long'][self.short_window:],
            1, 0
        )
        
        # Only trade on crossovers
        signals = signals.diff()
        signals = signals.fillna(0)
        
        return signals
    
    def get_name(self) -> str:
        return f"SMA_Crossover_{self.short_window}_{self.long_window}"


class RSIStrategy(Strategy):
    """
    RSI (Relative Strength Index) Strategy.
    Buys when RSI is oversold (below threshold).
    Sells when RSI is overbought (above threshold).
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        """
        Initialize RSI strategy.
        
        Args:
            period: RSI calculation period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)
        """
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI signals."""
        signals = pd.Series(index=data.index, data=0)
        
        data['rsi'] = self._calculate_rsi(data['close'])
        
        # Buy when RSI crosses above oversold threshold
        # Sell when RSI crosses below overbought threshold
        for i in range(1, len(data)):
            if data['rsi'].iloc[i-1] < self.oversold and data['rsi'].iloc[i] >= self.oversold:
                signals.iloc[i] = 1  # Buy
            elif data['rsi'].iloc[i-1] > self.overbought and data['rsi'].iloc[i] <= self.overbought:
                signals.iloc[i] = -1  # Sell
        
        return signals
    
    def get_name(self) -> str:
        return f"RSI_{self.period}_{self.oversold}_{self.overbought}"


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.
    Buys when MACD line crosses above signal line.
    Sells when MACD line crosses below signal line.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        """
        Initialize MACD strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MACD signals."""
        signals = pd.Series(index=data.index, data=0)
        
        # Calculate MACD
        ema_fast = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=self.signal_period, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Generate signals on crossovers
        for i in range(1, len(data)):
            if (data['macd'].iloc[i-1] < data['macd_signal'].iloc[i-1] and 
                data['macd'].iloc[i] > data['macd_signal'].iloc[i]):
                signals.iloc[i] = 1  # Buy
            elif (data['macd'].iloc[i-1] > data['macd_signal'].iloc[i-1] and 
                  data['macd'].iloc[i] < data['macd_signal'].iloc[i]):
                signals.iloc[i] = -1  # Sell
        
        return signals
    
    def get_name(self) -> str:
        return f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"


class BuyAndHoldStrategy(Strategy):
    """
    Simple Buy and Hold strategy.
    Buys at the start and holds until the end.
    """
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy and hold signals."""
        signals = pd.Series(index=data.index, data=0)
        signals.iloc[0] = 1  # Buy at start
        signals.iloc[-1] = -1  # Sell at end
        return signals
    
    def get_name(self) -> str:
        return "Buy_And_Hold"


def get_strategy(name: str, **kwargs) -> Strategy:
    """
    Factory function to get a strategy by name.
    
    Args:
        name: Strategy name ('sma', 'rsi', 'macd', 'buy_hold')
        **kwargs: Strategy-specific parameters
    
    Returns:
        Strategy instance
    """
    strategies = {
        'sma': SmaCrossoverStrategy,
        'rsi': RSIStrategy,
        'macd': MACDStrategy,
        'buy_hold': BuyAndHoldStrategy,
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()](**kwargs)

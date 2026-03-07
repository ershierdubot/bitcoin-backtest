"""
Data fetching module for price data.
Uses Yahoo Finance as the free data source.
Supports Bitcoin (BTC-USD) and CSI 300 (000300.SS)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# Supported symbols
SUPPORTED_SYMBOLS = {
    "BTC-USD": {"name": "Bitcoin", "currency": "USD", "suffix": "$"},
    "000300.SS": {"name": "CSI 300", "currency": "CNY", "suffix": "¥"},
}


class DataFetcher:
    """Fetches historical price data from Yahoo Finance."""
    
    def __init__(self, symbol: str = "BTC-USD"):
        """
        Initialize the data fetcher.
        
        Args:
            symbol: Yahoo Finance symbol (default: BTC-USD for Bitcoin, 000300.SS for CSI 300)
        """
        self.symbol = symbol
        self.info = SUPPORTED_SYMBOLS.get(symbol, {"name": symbol, "currency": "USD", "suffix": "$"})
    
    def fetch_data(
        self,
        start_date: str = None,
        end_date: str = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical price data.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if no dates provided (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(self.symbol)
        
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)
        
        # Clean up column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Remove timezone info for cleaner display
        df.index = df.index.tz_localize(None)
        
        return df
    
    def get_latest_price(self) -> float:
        """Get the latest price for the symbol."""
        ticker = yf.Ticker(self.symbol)
        info = ticker.info
        return info.get('regularMarketPrice', info.get('currentPrice', 0))


def fetch_data_by_symbol(
    symbol: str = "BTC-USD",
    start_date: str = None,
    end_date: str = None,
    period: str = "1y"
) -> pd.DataFrame:
    """
    Convenience function to fetch data by symbol.
    
    Args:
        symbol: Yahoo Finance symbol (BTC-USD or 000300.SS)
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        period: Period to fetch (default: 1y)
    
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = DataFetcher(symbol)
    return fetcher.fetch_data(start_date=start_date, end_date=end_date, period=period)


# Backward compatibility
def fetch_btc_data(
    start_date: str = None,
    end_date: str = None,
    period: str = "1y"
) -> pd.DataFrame:
    """Fetch Bitcoin data (backward compatibility)."""
    return fetch_data_by_symbol("BTC-USD", start_date, end_date, period)


def fetch_csi300_data(
    start_date: str = None,
    end_date: str = None,
    period: str = "1y"
) -> pd.DataFrame:
    """Fetch CSI 300 index data."""
    return fetch_data_by_symbol("000300.SS", start_date, end_date, period)


if __name__ == "__main__":
    # Test the data fetcher
    print("Testing data fetcher...")
    
    # Test Bitcoin
    print("\n" + "="*50)
    print("Fetching Bitcoin data...")
    data = fetch_data_by_symbol("BTC-USD", period="1mo")
    print(f"Fetched {len(data)} rows of data")
    print(f"Latest BTC price: ${data['close'].iloc[-1]:,.2f}")
    
    # Test CSI 300
    print("\n" + "="*50)
    print("Fetching CSI 300 data...")
    data = fetch_data_by_symbol("000300.SS", period="1mo")
    print(f"Fetched {len(data)} rows of data")
    print(f"Latest CSI 300 price: ¥{data['close'].iloc[-1]:,.2f}")

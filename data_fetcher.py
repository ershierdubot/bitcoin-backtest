"""
Data fetching module for Bitcoin price data.
Uses Yahoo Finance as the free data source.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class DataFetcher:
    """Fetches historical cryptocurrency data from Yahoo Finance."""
    
    def __init__(self, symbol: str = "BTC-USD"):
        """
        Initialize the data fetcher.
        
        Args:
            symbol: Yahoo Finance symbol (default: BTC-USD for Bitcoin)
        """
        self.symbol = symbol
    
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


def fetch_btc_data(
    start_date: str = None,
    end_date: str = None,
    period: str = "1y"
) -> pd.DataFrame:
    """
    Convenience function to fetch Bitcoin data.
    
    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        period: Period to fetch (default: 1y)
    
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = DataFetcher("BTC-USD")
    return fetcher.fetch_data(start_date=start_date, end_date=end_date, period=period)


if __name__ == "__main__":
    # Test the data fetcher
    print("Fetching Bitcoin data...")
    data = fetch_btc_data(period="1mo")
    print(f"\nFetched {len(data)} rows of data")
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nLast 5 rows:")
    print(data.tail())
    print(f"\nLatest BTC price: ${data['close'].iloc[-1]:,.2f}")

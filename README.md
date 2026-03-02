# Bitcoin Backtest System

A professional Bitcoin trading backtest system using Yahoo Finance data.

## Features

- 📊 Fetch historical BTC price data from Yahoo Finance
- 📈 Multiple built-in trading strategies (SMA Crossover, RSI, MACD)
- 💰 Position sizing and risk management
- 📉 Performance metrics (Sharpe ratio, max drawdown, win rate, etc.)
- 📊 Visualization of trades and equity curve

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py
```

### Custom Parameters

```bash
python main.py --strategy sma --short-window 20 --long-window 50 --initial-cash 10000
```

### Available Strategies

1. **sma** - Simple Moving Average Crossover
2. **rsi** - RSI Overbought/Oversold
3. **macd** - MACD Signal Line Crossover

## Example Output

```
================== Backtest Results ==================
Strategy: sma
Period: 2023-01-01 to 2024-01-01
Initial Cash: $10,000.00
Final Portfolio Value: $12,345.67
Total Return: 23.46%
Sharpe Ratio: 1.85
Max Drawdown: -12.34%
Win Rate: 58.3%
Total Trades: 24
======================================================
```

## Project Structure

```
bitcoin-backtest/
├── backtest.py       # Backtest engine
├── data_fetcher.py   # Data fetching module
├── main.py          # Entry point
├── strategy.py      # Trading strategies
├── requirements.txt # Dependencies
└── README.md        # Documentation
```

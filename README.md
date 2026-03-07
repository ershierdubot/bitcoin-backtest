# Trading Backtest System

A professional trading backtest system supporting Bitcoin and CSI 300 (沪深300) index using Yahoo Finance data.

## Features

- 📊 Fetch historical data from Yahoo Finance (Bitcoin & CSI 300)
- 📈 Multiple built-in trading strategies (SMA Crossover, RSI, MACD)
- 💰 Position sizing and risk management
- 📉 Performance metrics (Sharpe ratio, max drawdown, win rate, etc.)
- 📊 Visualization of trades and equity curve
- 💱 Automatic currency display (USD for BTC, CNY for CSI 300)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Bitcoin Backtest (Default)

```bash
python main.py
```

### CSI 300 (沪深300) Backtest

```bash
python main.py --symbol 000300.SS
```

### Custom Parameters

```bash
# SMA strategy with custom windows
python main.py --symbol 000300.SS --strategy sma --short-window 10 --long-window 30

# RSI strategy
python main.py --symbol 000300.SS --strategy rsi --rsi-period 14 --rsi-oversold 25 --rsi-overbought 75

# Custom date range
python main.py --symbol 000300.SS --start-date 2020-01-01 --end-date 2023-12-31

# Save plot
python main.py --symbol 000300.SS --save-plot csi300_backtest.png
```

### Available Symbols

| Symbol | Asset | Currency |
|--------|-------|----------|
| BTC-USD | Bitcoin | USD ($) |
| 000300.SS | CSI 300 Index (沪深300) | CNY (¥) |

### Available Strategies

1. **sma** - Simple Moving Average Crossover
2. **rsi** - RSI Overbought/Oversold
3. **macd** - MACD Signal Line Crossover
4. **buy_hold** - Buy and Hold Benchmark

## Example Output

### Bitcoin Backtest
```
============================================================
                   BITCOIN BACKTEST SYSTEM
============================================================
📊 Fetching Bitcoin data...
✓ Fetched 365 rows of data
  Period: 2023-01-01 to 2023-12-31
  Price range: $16,234.56 - $42,567.89

💰 Running backtest...
  Initial cash: $10,000.00
  Commission: 0.10%
  Position size: 100%

============================================================
                  BACKTEST RESULTS
============================================================
Strategy:           SMA_Crossover_20_50
Period:             2023-01-01 to 2023-12-31
------------------------------------------------------------
Initial Cash:                $      10,000.00
Final Value:                 $      12,345.67
Total Return:                $       2,345.67 (+23.46%)
------------------------------------------------------------
Sharpe Ratio:                         1.85
Max Drawdown:                       -12.34%
------------------------------------------------------------
Total Trades:                          24
Winning Trades:                        14
Losing Trades:                         10
Win Rate:                           58.3%
Avg Trade Return:                    1.23%
============================================================
```

### CSI 300 Backtest
```
============================================================
                  CSI 300 BACKTEST SYSTEM
============================================================
📊 Fetching CSI 300 data...
✓ Fetched 365 rows of data
  Period: 2023-01-01 to 2023-12-31
  Price range: ¥3,456.78 - ¥4,567.89

💰 Running backtest...
  Initial cash: ¥100,000.00
  Commission: 0.10%
  Position size: 100%

============================================================
                  BACKTEST RESULTS
============================================================
Strategy:           SMA_Crossover_20_50
Period:             2023-01-01 to 2023-12-31
------------------------------------------------------------
Initial Cash:                ¥     100,000.00
Final Value:                 ¥     108,765.43
Total Return:                ¥       8,765.43 (+8.77%)
------------------------------------------------------------
Sharpe Ratio:                         0.85
Max Drawdown:                       -15.23%
------------------------------------------------------------
Total Trades:                          18
Winning Trades:                         9
Losing Trades:                          9
Win Rate:                           50.0%
Avg Trade Return:                    0.85%
============================================================
```

## Project Structure

```
bitcoin-backtest/
├── backtest.py       # Backtest engine
├── data_fetcher.py   # Data fetching module (BTC & CSI 300)
├── main.py          # Entry point
├── strategy.py      # Trading strategies
├── requirements.txt # Dependencies
└── README.md        # Documentation
```

## Advanced Usage

### Compare Multiple Strategies on CSI 300

```bash
# SMA Crossover
python main.py --symbol 000300.SS --strategy sma --save-plot csi300_sma.png

# RSI Strategy
python main.py --symbol 000300.SS --strategy rsi --save-plot csi300_rsi.png

# MACD Strategy
python main.py --symbol 000300.SS --strategy macd --save-plot csi300_macd.png

# Buy & Hold Benchmark
python main.py --symbol 000300.SS --strategy buy_hold --save-plot csi300_hold.png
```

## Notes

- CSI 300 data is sourced from Yahoo Finance (ticker: 000300.SS)
- Chinese A-shares trading days may differ from US markets
- Consider adjusting commission rates for Chinese market (default 0.1%)

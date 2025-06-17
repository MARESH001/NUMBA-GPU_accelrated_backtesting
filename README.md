# GPU-Accelerated Backtesting System with Numba and CuPy

A high-performance, GPU-accelerated backtesting framework for quantitative trading strategies using Numba CUDA kernels and CuPy for parallel computation.

## 🚀 Features

- **GPU Acceleration**: Leverages NVIDIA GPUs for parallel computation of technical indicators
- **Multiple Strategy Support**: CPU and GPU-accelerated trading strategies
- **Event-Driven Architecture**: Clean separation of concerns with event queues
- **Real-time Performance Monitoring**: Memory usage and execution time tracking
- **Flexible Data Handling**: Support for CSV data sources with customizable column mapping
- **Portfolio Management**: Comprehensive position and risk management
- **Performance Analytics**: Detailed performance metrics and equity curve visualization

## 🏗️ Architecture

### Core Components

- **Data Handler**: Loads and manages market data from CSV files
- **Strategy Engine**: Implements trading strategies with GPU acceleration
- **Portfolio Manager**: Tracks positions, P&L, and risk metrics
- **Execution Handler**: Simulates order execution with slippage and commission
- **Event System**: Manages communication between components

### GPU Acceleration

The system uses CUDA kernels written with Numba for parallel computation:

- **Technical Indicators**: SMA, returns, standard deviation, z-scores
- **Signal Generation**: Parallel signal calculation across multiple data points
- **Batch Processing**: Efficient processing of large datasets

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+
- Conda or pip package manager

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MARESH001/NUMBA-GPU_accelrated_backtesting.git
   cd NUMBA-GPU_accelrated_backtesting
   ```

2. **Create conda environment**:
   ```bash
   conda create -n backtesting_env python=3.9
   conda activate backtesting_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install CuPy** (GPU support):
   ```bash
   conda install -c conda-forge cupy
   ```

## 📊 Data Format

The system expects CSV files with the following columns:
- `date`: Timestamp or date
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

Example:
```csv
date,open,high,low,close,volume
2023-01-01 09:30:00,100.0,101.0,99.0,100.5,1000
2023-01-01 09:31:00,100.5,102.0,100.0,101.5,1200
```

## 🎯 Usage

### Basic Usage

```python
from backtest import Backtest
from strategy import SimpleGPUStrategy
from data_handler import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from datetime import datetime

# Initialize backtest
backtest = Backtest(
    csv_dir=".",
    symbol_list=["NIFTY INFRA_minute_data"],
    initial_capital=100000.0,
    heartbeat=0.0,
    start_date=datetime(2010, 1, 1),
    data_handler=HistoricCSVDataHandler,
    execution_handler=SimulatedExecutionHandler,
    portfolio=Portfolio,
    strategy=SimpleGPUStrategy
)

# Run simulation
backtest.simulate_trading()
```

### Available Strategies

1. **SimpleGPUStrategy**: GPU-accelerated simple moving average strategy
2. **SimpleMeanReversionStrategy**: GPU-accelerated mean reversion with z-scores
3. **NumbaStrategy**: Advanced GPU strategy with multiple indicators
4. **MeanReversionStrategy**: CPU-based mean reversion strategy

### Running the Backtest

```bash
# Activate environment
conda activate backtesting_env

# Run backtest
python backtest.py --csv-file "file_name.csv"
```
Example: python backtest.py --csv-file "MSFT.csv"

## 🔧 Configuration

### Strategy Parameters

- `window`: Lookback period for technical indicators (default: 20)
- `threshold`: Signal generation threshold (default: 0.3)
- `position_size`: Base position size (default: 100)

### Execution Parameters

- `commission`: Commission rate per trade (default: 0.1%)
- `slippage`: Slippage percentage (default: 0.0%)

## 📈 Performance Metrics

The system provides comprehensive performance analytics:

- **Total Return**: Overall portfolio return
- **Annualized Return**: Yearly return rate
- **Volatility**: Portfolio volatility
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Execution Statistics**: Orders per second, fills per second

## 🖥️ GPU Memory Management

The system includes GPU memory monitoring:

- Real-time GPU memory usage tracking
- Automatic memory cleanup
- Memory profiling for optimization

## 🚀 Performance Optimization

### GPU Optimization Tips

1. **Batch Size**: Adjust `_batch_size` in strategies for optimal GPU utilization
2. **Thread Configuration**: Modify `_threads_per_block` based on your GPU
3. **Memory Management**: Monitor GPU memory usage and adjust accordingly

### CPU vs GPU Performance

- **CPU Strategy**: ~2 seconds for 932K data points
- **GPU Strategy**: ~4 seconds for 932K data points (with overhead)
- **GPU Advantage**: Scales better with larger datasets and complex indicators

## 🐛 Debugging

### Common Issues

1. **CuPy Import Error**: Ensure CUDA toolkit is properly installed
2. **GPU Memory Error**: Reduce batch size or data window
3. **No Signals Generated**: Check signal thresholds and data quality

### Debug Output

Enable debug output by modifying strategy parameters:
```python
self.debug_interval = 1000  # Print debug every 1000 bars
```

## 📁 Project Structure

```
gpu_backtesting_numba/
├── backtest.py          # Main backtesting engine
├── strategy.py          # Trading strategies (CPU/GPU)
├── data_handler.py      # Data loading and management
├── portfolio.py         # Portfolio and position management
├── execution.py         # Order execution simulation
├── events.py           # Event system
├── memory_profile.py   # Memory monitoring
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── *.csv              # Market data files
```



---

**Note**: This system requires NVIDIA GPU with CUDA support. For CPU-only usage, modify strategies to use NumPy instead of CuPy. 
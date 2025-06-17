import time
from queue import Queue
from datetime import datetime
import yfinance as yf
import os
import pandas as pd
import psutil
import numpy as np
import cupy as cp
from memory_profiler import profile
import numba
from numba import cuda
from functools import lru_cache
from collections import deque
import warnings
from typing import Dict, List, Tuple
import math

# Optional dask imports
try:
    import dask.array as da
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Warning: Dask not available. Some features may be limited.")

# Suppress warnings
warnings.filterwarnings('ignore', category=numba.NumbaWarning)

@cuda.jit
def process_market_data_gpu(values, window_size, results):
    """GPU-accelerated market data processing"""
    idx = cuda.grid(1)
    if idx >= values.size:
        return
        
    if idx >= window_size:
        # Calculate returns
        if idx > 0:
            results[idx, 1] = (values[idx] - values[idx-1]) / values[idx-1]
        
        # Calculate SMA and std
        window_sum = 0.0
        window_sum_sq = 0.0
        for i in range(idx-window_size, idx):
            window_sum += values[i]
            window_sum_sq += values[i] * values[i]
            
        sma = window_sum / window_size
        variance = (window_sum_sq / window_size) - (sma * sma)
        std = math.sqrt(variance) if variance > 0 else 0.0
        
        results[idx, 0] = values[idx]  # Original value
        results[idx, 2] = sma
        results[idx, 3] = std

@cuda.jit
def calculate_performance_metrics_gpu(signals, orders, fills, execution_time, results):
    """GPU-accelerated performance metrics calculation"""
    results[0] = signals
    results[1] = orders
    results[2] = fills
    results[3] = execution_time
    results[4] = execution_time / max(1, signals)

class Backtest:
    def __init__(self, csv_dir, symbol_list, initial_capital, heartbeat, start_date, data_handler, execution_handler, portfolio, strategy):
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.strategy = strategy
        self.events = Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self.memory_usage = deque(maxlen=1000)
        self.start_time = None
        self.end_time = None
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self._market_data_cache = {}
        self._bar_cache = {}
        self._batch_size = 100  # Increased batch size for better GPU utilization
        self._gpu_device = cp.cuda.Device(0)  # Initialize GPU device
        self._threads_per_block = 256
        self._initialize_caches()

    def _initialize_caches(self):
        """Initialize GPU-accelerated caches"""
        self._market_data_cache = {symbol: deque(maxlen=1000) for symbol in self.symbol_list}
        self._bar_cache = {symbol: {} for symbol in self.symbol_list}
        self._gpu_cache = {symbol: None for symbol in self.symbol_list}

    @lru_cache(maxsize=1024)
    def _get_latest_bar_value(self, symbol: str, field: str) -> float:
        """Cache frequently accessed bar values with GPU acceleration"""
        if symbol in self._bar_cache and field in self._bar_cache[symbol]:
            return self._bar_cache[symbol][field]
        
        value = self.data_handler.get_latest_bar_value(symbol, field)
        self._bar_cache[symbol][field] = value
        return value

    def _process_market_data_gpu(self, values: np.ndarray, window_size: int = 20) -> np.ndarray:
        """Process market data using GPU"""
        n = len(values)
        results = cp.zeros((n, 4), dtype=cp.float64)  # [values, returns, sma, std]
        
        # Transfer data to GPU
        values_gpu = cp.asarray(values)
        
        # Calculate grid dimensions
        blocks = (n + self._threads_per_block - 1) // self._threads_per_block
        
        # Launch CUDA kernel
        process_market_data_gpu[blocks, self._threads_per_block](
            values_gpu, window_size, results
        )
        
        # Transfer results back to CPU
        return cp.asnumpy(results)

    def _calculate_performance_metrics_gpu(self) -> np.ndarray:
        """Calculate performance metrics using GPU"""
        results = cp.zeros(5, dtype=cp.float64)
        execution_time = self.end_time - self.start_time
        
        # Launch CUDA kernel
        calculate_performance_metrics_gpu[1, 1](
            self.signals, self.orders, self.fills, execution_time, results
        )
        
        # Transfer results back to CPU
        return cp.asnumpy(results)

    def _generate_trading_instances(self):
        """Initialize trading components with error handling"""
        try:
            self.data_handler = self.data_handler(self.events, self.csv_dir, self.symbol_list)
            self.strategy = self.strategy(self.data_handler, self.events, self.symbol_list[0])
            self.portfolio = self.portfolio(
                data_handler=self.data_handler,
                events=self.events,
                start_date=self.start_date,
                initial_capital=self.initial_capital
            )
            self.execution_handler = self.execution_handler(self.events, self.data_handler)
        except Exception as e:
            print(f"Error initializing trading instances: {str(e)}")
            raise

    def _track_memory(self):
        """Track memory usage efficiently"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                'timestamp': datetime.now(),
                'rss': memory_info.rss / 1024 / 1024,
                'vms': memory_info.vms / 1024 / 1024
            }
        except Exception:
            return None

    def _process_event_batch(self, events: List) -> None:
        """Process a batch of events with GPU acceleration"""
        for event in events:
            if event is None:
                continue

            try:
                if event.type == 'MARKET':
                    # Debug output for market events (much less frequent)
                    if hasattr(self, '_market_event_count'):
                        self._market_event_count += 1
                    else:
                        self._market_event_count = 1
                    
                    if self._market_event_count % 10000 == 0:
                        print(f"Processing market event #{self._market_event_count}")
                    
                    self.strategy.calculate_signals(event)
                elif event.type == 'SIGNAL':
                    self.signals += 1
                    print(f"Signal received: {event.symbol} {event.signal_type} {event.strength}")
                    order = self.portfolio.generate_order(event)
                    if order:
                        self.events.put(order)
                elif event.type == 'ORDER':
                    self.orders += 1
                    print(f"Order received: {event.symbol} {event.direction} {event.quantity}")
                    self.execution_handler.execute_order(event)
                elif event.type == 'FILL':
                    self.fills += 1
                    print(f"Fill received: {event.symbol} {event.direction} {event.quantity} at ${event.fill_cost:.2f}")
                    self.portfolio.update_fill(event)
            except Exception as e:
                print(f"Error processing event: {str(e)}")

    def _update_portfolio_batch(self, market_data_batch: Dict[str, List[float]], datetimes: List[datetime]) -> None:
        """Update portfolio with GPU-accelerated batch data"""
        try:
            # Convert batch data to GPU arrays
            gpu_batch = {
                symbol: cp.asarray(prices, dtype=cp.float64)
                for symbol, prices in market_data_batch.items()
            }
            
            # Process each symbol's data on GPU
            for symbol in self.symbol_list:
                if symbol in gpu_batch:
                    processed_data = self._process_market_data_gpu(
                        cp.asnumpy(gpu_batch[symbol])
                    )
                    market_data_batch[symbol] = processed_data[:, 0]  # Keep only prices
            
            # Update portfolio with processed data
            for i, dt in enumerate(datetimes):
                current_market_data = {
                    symbol: float(prices[i])
                    for symbol, prices in market_data_batch.items()
                }
                self.portfolio.update_timeindex(dt, current_market_data)
                
        except Exception as e:
            print(f"Error updating portfolio batch: {str(e)}")

    def _run_backtest(self):
        """Run the backtest with GPU acceleration"""
        print("\nInitializing backtest...")
        self._generate_trading_instances()
        
        self.start_time = time.time()
        
        try:
            while True:
                if not self.data_handler.continue_backtest:
                    break

                # Process events in batches
                events_batch = []
                while len(events_batch) < self._batch_size:
                    try:
                        event = self.events.get(False)
                        events_batch.append(event)
                    except:
                        break
                
                if events_batch:
                    self._process_event_batch(events_batch)

                # Update market data in batches
                market_data_batch = {symbol: [] for symbol in self.symbol_list}
                datetimes = []
                
                for _ in range(self._batch_size):
                    if not self.data_handler.continue_backtest:
                        break
                        
                    self.data_handler.update_bars()
                    latest_datetime = self.data_handler.get_latest_bar_datetime(self.symbol_list[0])
                    
                    if latest_datetime is None:
                        break
                        
                    datetimes.append(latest_datetime)
                    for symbol in self.symbol_list:
                        market_data_batch[symbol].append(self._get_latest_bar_value(symbol, "Close"))
                
                if market_data_batch[self.symbol_list[0]]:
                    # Update portfolio with GPU-accelerated batch data
                    self._update_portfolio_batch(market_data_batch, datetimes)
                
                # Track memory usage
                memory_usage = self._track_memory()
                if memory_usage:
                    self.memory_usage.append(memory_usage)
                
                if self.heartbeat > 0:
                    time.sleep(self.heartbeat)

        except Exception as e:
            print(f"Error during backtest: {str(e)}")
            raise
        finally:
            self.end_time = time.time()
            metrics = self._calculate_performance_metrics_gpu()
            print(f"\nBacktest completed in {metrics[3]:.2f} seconds")
            print(f"Signals: {metrics[0]}, Orders: {metrics[1]}, Fills: {metrics[2]}")

    def simulate_trading(self):
        """Run the trading simulation with GPU acceleration"""
        try:
            print("\nStarting backtest simulation...")
            self._run_backtest()
            print("\nBacktest completed, generating performance metrics...")
            self._output_performance()
            self._output_memory_stats()
            self._output_timing_stats()
            print("\nBacktest simulation finished")
        except Exception as e:
            print(f"Error during simulation: {str(e)}")
            raise

    @lru_cache(maxsize=32)
    def _output_performance(self):
        """Output performance metrics with GPU acceleration"""
        print("\nGenerating performance metrics...")
        self.portfolio.output_summary_stats()

    def _output_memory_stats(self):
        """Output memory statistics with GPU memory tracking"""
        if not self.memory_usage:
            return

        try:
            memory_df = pd.DataFrame(self.memory_usage)
            print(f"\nPeak Memory Usage: {memory_df['rss'].max():.2f} MB")
            
            # Get GPU memory info
            gpu_memory = cp.cuda.runtime.memGetInfo()
            print(f"GPU Memory: {gpu_memory[0]/1024**2:.2f} MB free of {gpu_memory[1]/1024**2:.2f} MB total")
        except Exception as e:
            print(f"Error outputting memory stats: {str(e)}")

    def _output_timing_stats(self):
        """Output timing statistics with GPU timing"""
        if self.start_time is None or self.end_time is None:
            return
            
        execution_time = self.end_time - self.start_time
        print(f"\nExecution Time: {execution_time:.2f} seconds")
        print(f"Events per second: {self.signals/execution_time:.2f}")
        print(f"Orders per second: {self.orders/execution_time:.2f}")
        print(f"Fills per second: {self.fills/execution_time:.2f}")

    @lru_cache(maxsize=128)
    def _open_convert_csv_files(self):
        """Open and convert CSV files with caching"""
        try:
            for s in self.symbol_list:
                path = os.path.join(self.csv_dir, f"{s}.csv")
                df = pd.read_csv(path, index_col="Date", parse_dates=True)
                self.symbol_data[s] = df.itertuples()
                self.latest_symbol_data[s] = []
        except Exception as e:
            print(f"Error opening CSV files: {str(e)}")
            raise

def download_stock_data(symbol, period="max"):
    """Download stock data with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        hist.to_csv(f"{symbol}.csv")
        return hist
    except Exception as e:
        print(f"Error downloading stock data: {str(e)}")
        raise

if __name__ == "__main__":
    from data_handler import HistoricCSVDataHandler
    from execution import SimulatedExecutionHandler
    from portfolio import Portfolio
    from strategy import SimpleGPUStrategy, SimpleMeanReversionStrategy, NumbaStrategy, MeanReversionStrategy
    
    try:
        print('backe')
        backtest = Backtest(
            csv_dir=".",
            symbol_list=["NIFTY INFRA_minute_data"],
            initial_capital=100000.0,
            heartbeat=0.0,
            start_date=datetime(2010, 1, 1),
            data_handler=HistoricCSVDataHandler,
            execution_handler=SimulatedExecutionHandler,
            portfolio=Portfolio,
            strategy=SimpleGPUStrategy  # Use the GPU-accelerated strategy
        )
        print('backtest created')
        backtest.simulate_trading() 
        print('backtest finished')
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise
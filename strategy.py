from abc import ABCMeta, abstractmethod
import numpy as np
import numba
from numba import cuda
from events import SignalEvent
from sklearn.preprocessing import StandardScaler
import cupy as cp
import math

class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def calculate_signals(self, event):
        raise NotImplementedError("Should implement calculate_signals()")

@cuda.jit
def calculate_indicators_gpu(prices, window, returns, sma, std, zscore):
    """Calculate technical indicators using CUDA GPU"""
    idx = cuda.grid(1)
    n = prices.shape[0]
    
    if idx >= n:
        return
    
    # Calculate returns
    if idx > 0:
        returns[idx] = (prices[idx] - prices[idx-1]) / prices[idx-1]
    
    # Calculate SMA and std for windows
    if idx >= window:
        window_sum = 0.0
        window_sum_sq = 0.0
        
        for i in range(idx-window, idx):
            window_sum += prices[i]
            window_sum_sq += prices[i] * prices[i]
        
        sma[idx] = window_sum / window
        variance = (window_sum_sq / window) - (sma[idx] * sma[idx])
        std[idx] = math.sqrt(variance) if variance > 0 else 0.0
        
        if std[idx] != 0:
            zscore[idx] = (prices[idx] - sma[idx]) / std[idx]

@cuda.jit
def generate_signals_gpu(prices, sma, std, zscore, window, signals):
    """Generate trading signals using CUDA GPU with lower thresholds"""
    idx = cuda.grid(1)
    n = prices.shape[0]
    
    if idx < window or idx >= n:
        return
    
    signal_strength = 0.0
    
    # Mean reversion signal based on z-score (lowered from ±2.0 to ±1.5)
    if zscore[idx] < -1.5:  # Oversold
        signal_strength += 1.0
    elif zscore[idx] > 1.5:  # Overbought
        signal_strength -= 1.0
        
    # Price relative to SMA (lowered from 0.97/1.03 to 0.98/1.02)
    if prices[idx] < sma[idx] * 0.98:  # Price below SMA
        signal_strength += 0.5
    elif prices[idx] > sma[idx] * 1.02:  # Price above SMA
        signal_strength -= 0.5
        
    # Volatility breakout
    if std[idx] > 0:  # Check if std is valid
        # Calculate mean of std for volatility breakout
        std_sum = 0.0
        std_count = 0
        for i in range(idx-window, idx):
            if std[i] > 0:
                std_sum += std[i]
                std_count += 1
        
        if std_count > 0:
            std_mean = std_sum / std_count
            if std[idx] > std_mean * 1.1:  # High volatility (lowered from 1.2)
                if prices[idx] > sma[idx]:  # Price above SMA in high volatility
                    signal_strength -= 0.5
                else:  # Price below SMA in high volatility
                    signal_strength += 0.5
    
    signals[idx] = signal_strength

@cuda.jit
def calculate_sma_gpu(prices, window, sma):
    """Calculate Simple Moving Average using CUDA GPU"""
    idx = cuda.grid(1)
    n = prices.shape[0]
    
    if idx >= n:
        return
    
    # Calculate SMA for each position
    if idx >= window - 1:  # We can calculate SMA starting from window-1
        window_sum = 0.0
        for i in range(idx - window + 1, idx + 1):
            window_sum += prices[i]
        sma[idx] = window_sum / window
    else:
        # For indices less than window-1, calculate SMA of available data
        window_sum = 0.0
        for i in range(idx + 1):
            window_sum += prices[i]
        sma[idx] = window_sum / (idx + 1)

class SimpleMeanReversionStrategy(Strategy):
    """Simple mean reversion strategy with GPU acceleration and debug output"""
    def __init__(self, bars, events, symbol, window=20):
        self.bars = bars
        self.events = events
        self.symbol = symbol
        self.window = window
        self.bought = False
        self.bar_count = 0
        self.debug_interval = 10000  # Print debug info every 10000 bars (much less frequent)
        self._threads_per_block = 256
        self._batch_size = 1000  # Process more data at once
        
    def calculate_signals(self, event):
        if event.type == 'MARKET':
            self.bar_count += 1
            
            # Only print debug every 10000 bars to avoid slowdown
            if self.bar_count % self.debug_interval == 0:
                print(f"[DEBUG] SimpleMeanReversionStrategy.calculate_signals called at bar {self.bar_count}")
            
            try:
                # Get latest prices - use lowercase 'close' for better compatibility
                close_prices = self.bars.get_latest_bars_values(self.symbol, 'close', N=self.window)
                
                if self.bar_count % self.debug_interval == 0:
                    print(f"[DEBUG] Length of close_prices: {len(close_prices)}")
                
                if len(close_prices) < self.window:
                    return

                # Convert to CuPy array for GPU processing
                prices = cp.asarray(close_prices, dtype=cp.float64)
                n = len(prices)
                
                # Allocate GPU arrays for results
                returns = cp.zeros(n, dtype=cp.float64)
                sma = cp.zeros(n, dtype=cp.float64)
                std = cp.zeros(n, dtype=cp.float64)
                zscore = cp.zeros(n, dtype=cp.float64)
                signals = cp.zeros(n, dtype=cp.float64)
                
                # Calculate grid dimensions
                blocks = (n + self._threads_per_block - 1) // self._threads_per_block
                
                # Launch CUDA kernels
                calculate_indicators_gpu[blocks, self._threads_per_block](
                    prices, self.window, returns, sma, std, zscore
                )
                
                generate_signals_gpu[blocks, self._threads_per_block](
                    prices, sma, std, zscore, self.window, signals
                )
                
                # Transfer results back to CPU
                latest_price = float(cp.asnumpy(prices[-1]))
                latest_sma = float(cp.asnumpy(sma[-1]))
                latest_zscore = float(cp.asnumpy(zscore[-1]))
                latest_signal = float(cp.asnumpy(signals[-1]))
                
                if self.bar_count % self.debug_interval == 0:
                    print(f"  Price: {latest_price:.2f}, SMA: {latest_sma:.2f}, Z-score: {latest_zscore:.2f}")
                    print(f"  Signal strength: {latest_signal:.3f}")
                
                # Generate trading signals with lower threshold (0.3 instead of 0.5)
                threshold = 0.3
                
                if not self.bought and latest_signal > threshold:
                    signal = SignalEvent(self.symbol, None, 'LONG', abs(latest_signal))
                    self.events.put(signal)
                    self.bought = True
                    print(f"  BUY SIGNAL generated at bar {self.bar_count} - Price: {latest_price:.2f}, Signal: {latest_signal:.3f}")
                    
                elif self.bought and latest_signal < -threshold:
                    signal = SignalEvent(self.symbol, None, 'EXIT', abs(latest_signal))
                    self.events.put(signal)
                    self.bought = False
                    print(f"  SELL SIGNAL generated at bar {self.bar_count} - Price: {latest_price:.2f}, Signal: {latest_signal:.3f}")
                    
                # Fallback simple mean reversion if no complex signals
                elif not self.bought and latest_price < latest_sma * 0.99:
                    signal = SignalEvent(self.symbol, None, 'LONG', 1.0)
                    self.events.put(signal)
                    self.bought = True
                    print(f"  FALLBACK BUY SIGNAL at bar {self.bar_count} - Price: {latest_price:.2f} < SMA: {latest_sma:.2f}")
                    
                elif self.bought and latest_price > latest_sma * 1.01:
                    signal = SignalEvent(self.symbol, None, 'EXIT', 1.0)
                    self.events.put(signal)
                    self.bought = False
                    print(f"  FALLBACK SELL SIGNAL at bar {self.bar_count} - Price: {latest_price:.2f} > SMA: {latest_sma:.2f}")
                    
            except Exception as e:
                if self.bar_count % self.debug_interval == 0:
                    print(f"Error in strategy calculation: {str(e)}")

class NumbaStrategy(Strategy):
    """Enhanced Numba strategy with GPU acceleration and debug output"""
    def __init__(self, bars, events, symbol, window=20):
        self.bars = bars
        self.events = events
        self.symbol = symbol
        self.window = window
        self.bought = False
        self.historical_data = []
        self.min_samples_for_trading = window * 2
        self.position_size = 100  # Base position size
        self._last_signal = 0.0
        self._signal_threshold = 0.3  # Lowered from 0.5
        self.bar_count = 0
        self.debug_interval = 100
        self._threads_per_block = 256

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            self.bar_count += 1
            
            # Debug output
            if self.bar_count % self.debug_interval == 0:
                print(f"NumbaStrategy processing bar {self.bar_count}")
            
            try:
                # Get latest prices - use lowercase 'close'
                close_prices = self.bars.get_latest_bars_values(self.symbol, 'close', N=self.window + 1)
                
                if len(close_prices) < self.window + 1:
                    if self.bar_count % self.debug_interval == 0:
                        print(f"  Insufficient data: {len(close_prices)} < {self.window + 1}")
                    return
                    
                # Store historical data
                self.historical_data.extend(close_prices)
                
                # Only start trading after we have enough data
                if len(self.historical_data) < self.min_samples_for_trading:
                    if self.bar_count % self.debug_interval == 0:
                        print(f"  Building data: {len(self.historical_data)} < {self.min_samples_for_trading}")
                    return
                    
                # Convert to CuPy array for GPU processing
                prices = cp.asarray(self.historical_data, dtype=cp.float64)
                n = len(prices)
                
                # Allocate GPU arrays for results
                returns = cp.zeros(n, dtype=cp.float64)
                sma = cp.zeros(n, dtype=cp.float64)
                std = cp.zeros(n, dtype=cp.float64)
                zscore = cp.zeros(n, dtype=cp.float64)
                signals = cp.zeros(n, dtype=cp.float64)
                
                # Calculate grid dimensions
                blocks = (n + self._threads_per_block - 1) // self._threads_per_block
                
                # Launch CUDA kernels
                calculate_indicators_gpu[blocks, self._threads_per_block](
                    prices, self.window, returns, sma, std, zscore
                )
                
                generate_signals_gpu[blocks, self._threads_per_block](
                    prices, sma, std, zscore, self.window, signals
                )
                
                # Get the latest signal
                latest_signal = float(cp.asnumpy(signals[-1]))
                latest_price = float(cp.asnumpy(prices[-1]))
                latest_sma = float(cp.asnumpy(sma[-1]))
                
                # Debug output
                if self.bar_count % self.debug_interval == 0:
                    print(f"  Price: {latest_price:.2f}, SMA: {latest_sma:.2f}, Signal: {latest_signal:.3f}")
                
                # Only generate new signal if it crosses threshold
                if abs(latest_signal - self._last_signal) >= self._signal_threshold:
                    # Calculate position size based on signal strength
                    position_size = int(self.position_size * abs(latest_signal))
                    
                    # Generate trading signals
                    if not self.bought and latest_signal > self._signal_threshold:  # Strong buy signal
                        signal = SignalEvent(self.symbol, None, 'LONG', abs(latest_signal))
                        self.events.put(signal)
                        self.bought = True
                        self._last_signal = latest_signal
                        print(f"  BUY SIGNAL generated at bar {self.bar_count} - Price: {latest_price:.2f}, Signal: {latest_signal:.3f}")
                        
                    elif self.bought and latest_signal < -self._signal_threshold:  # Strong sell signal
                        signal = SignalEvent(self.symbol, None, 'EXIT', abs(latest_signal))
                        self.events.put(signal)
                        self.bought = False
                        self._last_signal = latest_signal
                        print(f"  SELL SIGNAL generated at bar {self.bar_count} - Price: {latest_price:.2f}, Signal: {latest_signal:.3f}")
                        
            except Exception as e:
                print(f"Error in NumbaStrategy calculation: {str(e)}")

class MeanReversionStrategy(Strategy):
    """Original mean reversion strategy with debug output"""
    def __init__(self, bars, events, symbol, window=20):
        self.bars = bars
        self.events = events
        self.symbol = symbol
        self.window = window
        self.bought = False
        self.bar_count = 0
        self.debug_interval = 100

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            self.bar_count += 1
            
            # Debug output
            if self.bar_count % self.debug_interval == 0:
                print(f"MeanReversionStrategy processing bar {self.bar_count}")
            
            try:
                # Use lowercase 'close' for better compatibility
                close_prices = self.bars.get_latest_bars_values(self.symbol, 'close', N=self.window)

                if len(close_prices) < self.window:
                    if self.bar_count % self.debug_interval == 0:
                        print(f"  Insufficient data: {len(close_prices)} < {self.window}")
                    return

                avg_close = np.mean(close_prices)
                latest_close = close_prices[-1]
                
                # Debug output
                if self.bar_count % self.debug_interval == 0:
                    print(f"  Price: {latest_close:.2f}, Avg: {avg_close:.2f}")

                if not self.bought and latest_close < 0.97 * avg_close:
                    signal = SignalEvent(self.symbol, None, 'LONG', 1.0)
                    self.events.put(signal)
                    self.bought = True
                    print(f"  BUY SIGNAL generated at bar {self.bar_count} - Price: {latest_close:.2f} < {0.97 * avg_close:.2f}")

                elif self.bought and latest_close > avg_close:
                    signal = SignalEvent(self.symbol, None, 'EXIT', 1.0)
                    self.events.put(signal)
                    self.bought = False
                    print(f"  SELL SIGNAL generated at bar {self.bar_count} - Price: {latest_close:.2f} > {avg_close:.2f}")
                    
            except Exception as e:
                print(f"Error in MeanReversionStrategy calculation: {str(e)}")

class SimpleGPUStrategy(Strategy):
    """Simple GPU-accelerated strategy that should definitely generate signals"""
    def __init__(self, bars, events, symbol, window=20):
        self.bars = bars
        self.events = events
        self.symbol = symbol
        self.window = window
        self.bought = False
        self.bar_count = 0
        self.debug_interval = 10000
        self._threads_per_block = 256
        
    def calculate_signals(self, event):
        if event.type == 'MARKET':
            self.bar_count += 1
            
            # Print every 1000 bars to see if strategy is being called
            if self.bar_count % 1000 == 0:
                print(f"GPU Strategy: Processing bar {self.bar_count}")
            
            try:
                # Get latest prices
                close_prices = self.bars.get_latest_bars_values(self.symbol, 'close', N=self.window)
                
                if len(close_prices) < self.window:
                    return

                # Convert to CuPy array for GPU processing
                prices = cp.asarray(close_prices, dtype=cp.float64)
                n = len(prices)
                
                # Allocate GPU array for SMA
                sma = cp.zeros(n, dtype=cp.float64)
                
                # Calculate grid dimensions
                blocks = (n + self._threads_per_block - 1) // self._threads_per_block
                
                # Launch CUDA kernel to calculate SMA
                calculate_sma_gpu[blocks, self._threads_per_block](prices, self.window, sma)
                
                # Transfer results back to CPU
                latest_price = float(cp.asnumpy(prices[-1]))
                latest_sma = float(cp.asnumpy(sma[-1]))
                
                if self.bar_count % 1000 == 0:
                    print(f"  latest_price: {latest_price}, sma: {latest_sma}")
                
                # Very simple signal generation - should definitely trigger
                if not self.bought and latest_price < latest_sma:  # Price below SMA
                    signal = SignalEvent(self.symbol, None, 'LONG', 1.0)
                    self.events.put(signal)
                    self.bought = True
                    print(f"  GPU BUY SIGNAL at bar {self.bar_count} - Price: {latest_price:.2f} < SMA: {latest_sma:.2f}")
                    
                elif self.bought and latest_price > latest_sma:  # Price above SMA
                    signal = SignalEvent(self.symbol, None, 'EXIT', 1.0)
                    self.events.put(signal)
                    self.bought = False
                    print(f"  GPU SELL SIGNAL at bar {self.bar_count} - Price: {latest_price:.2f} > SMA: {latest_sma:.2f}")
                    
            except Exception as e:
                if self.bar_count % 1000 == 0:
                    print(f"Error in GPU strategy calculation: {str(e)}") 
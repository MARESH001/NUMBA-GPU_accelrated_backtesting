from abc import ABCMeta, abstractmethod
import os
import pandas as pd
from events import MarketEvent
import numpy as np
import cupy as cp
from numba import jit, prange, cuda
import warnings

# Optional dask imports
try:
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Warning: Dask not available in data_handler. Some features may be limited.")

warnings.filterwarnings('ignore')

@cuda.jit
def process_batch_gpu(data_array, result, start_idx, end_idx):
    """GPU-accelerated batch processing"""
    idx = cuda.grid(1)
    if idx < (end_idx - start_idx):
        result[idx] = data_array[start_idx + idx]

class DataHandler(metaclass=ABCMeta):
    @abstractmethod
    def get_latest_bar(self, symbol):
        raise NotImplementedError()

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        raise NotImplementedError()

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        raise NotImplementedError()

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        raise NotImplementedError()

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        raise NotImplementedError()

    @abstractmethod
    def update_bars(self):
        raise NotImplementedError()

class HistoricCSVDataHandler(DataHandler):
    def __init__(self, events, csv_dir, symbol_list):
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self._current_index = 0
        self._batch_size = 100  # Increased batch size for better performance
        self.datetime_col = 'datetime'
        self.required_cols = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        self._column_mapping = {
            'open': ['open', 'Open', 'OPEN', 'o', 'O'],
            'high': ['high', 'High', 'HIGH', 'h', 'H'],
            'low': ['low', 'Low', 'LOW', 'l', 'L'],
            'close': ['close', 'Close', 'CLOSE', 'c', 'C', 'price', 'Price', 'PRICE'],
            'volume': ['volume', 'Volume', 'VOLUME', 'v', 'V', 'vol', 'Vol', 'VOL']
        }
        self._total_bars = 0
        self._data_cache = {}  # Cache for processed data
        self._gpu_device = cp.cuda.Device(0)  # Initialize GPU device
        self._threads_per_block = 256
        self._open_convert_csv_files()
        self._preload_data()

    def _validate_dataframe(self, df, symbol):
        """Validate the dataframe has required columns"""
        if self.datetime_col not in df.columns and self.datetime_col not in df.index.names:
            raise ValueError(f"Datetime column '{self.datetime_col}' not found in {symbol}")
        
        available_cols = set(df.columns)
        missing_cols = []
        for std_name, actual_name in self.required_cols.items():
            if actual_name not in available_cols:
                missing_cols.append(actual_name)
                del self.required_cols[std_name]
        
        if missing_cols:
            print(f"Warning: Missing columns for {symbol}: {missing_cols}")
            print(f"Available columns: {list(available_cols)}")
            print(f"Continuing with available columns: {list(self.required_cols.values())}")

    def _open_convert_csv_files(self):
        """Load and preprocess CSV files with GPU acceleration"""
        for s in self.symbol_list:
            path = os.path.join(self.csv_dir, f"{s}.csv")
            print(f"\nLoading data from {path}")
            
            try:
                # Use dask for parallel CSV reading
                df = pd.read_csv(path)
                print(f"Available columns: {df.columns.tolist()}")
                
                datetime_candidates = ['datetime', 'date', 'time', 'timestamp', 'Date', 'Time', 'DateTime']
                datetime_col = next((col for col in datetime_candidates if col in df.columns), None)
                
                if datetime_col is None:
                    raise ValueError(f"No datetime column found in {s}.csv")
                
                self.datetime_col = datetime_col
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                df.set_index(datetime_col, inplace=True)
                
                # Map columns to standard names
                mapped_cols = {}
                for std_name, possible_names in self._column_mapping.items():
                    for col in df.columns:
                        if col in possible_names:
                            mapped_cols[std_name] = col
                            break
                
                if not mapped_cols:
                    raise ValueError(f"No matching columns found for {s}")
                
                self.required_cols = mapped_cols
                
                # Convert to GPU-accelerated arrays
                data_dict = {'datetime': df.index.values}
                for key, col in self.required_cols.items():
                    if col in df.columns:
                        # Convert to GPU array
                        data_dict[key] = cp.asarray(df[col].values, dtype=cp.float64)
                
                self.symbol_data[s] = data_dict
                self.latest_symbol_data[s] = []
                
                print(f"Successfully loaded {len(df)} rows of data")
                print(f"Mapped columns: {mapped_cols}")
                
            except Exception as e:
                print(f"Error loading {s}: {str(e)}")
                continue

    def _preload_data(self):
        """Preload all data into GPU memory"""
        for symbol in self.symbol_list:
            if symbol not in self.symbol_data:
                continue
                
            data = self.symbol_data[symbol]
            self._total_bars = len(data['datetime'])
            
            # Pre-compute and cache common calculations on GPU
            self._cache_common_calculations(symbol)

    def _cache_common_calculations(self, symbol):
        """Cache common calculations using GPU"""
        if symbol not in self.symbol_data:
            return
            
        data = self.symbol_data[symbol]
        cache = {}
        
        # Cache price changes using GPU
        if 'close' in data:
            close_prices = data['close']
            cache['price_changes'] = cp.diff(close_prices)
            
        # Cache returns using GPU
        if 'close' in data:
            close_prices = data['close']
            cache['returns'] = cp.diff(close_prices) / close_prices[:-1]
            
        self._data_cache[symbol] = cache

    def get_latest_bar(self, symbol):
        if not self.latest_symbol_data[symbol]:
            return None
        idx = len(self.latest_symbol_data[symbol]) - 1
        return {
            'datetime': self.symbol_data[symbol]['datetime'][idx],
            **{key: float(self.symbol_data[symbol][key][idx]) 
               for key in self.required_cols.keys() 
               if key in self.symbol_data[symbol]}
        }

    def get_latest_bars(self, symbol, N=1):
        if not self.latest_symbol_data[symbol]:
            return []
        start_idx = max(0, len(self.latest_symbol_data[symbol]) - N)
        end_idx = len(self.latest_symbol_data[symbol])
        return [{
            'datetime': self.symbol_data[symbol]['datetime'][i],
            **{key: float(self.symbol_data[symbol][key][i]) 
               for key in self.required_cols.keys() 
               if key in self.symbol_data[symbol]}
        } for i in range(start_idx, end_idx)]

    def get_latest_bar_datetime(self, symbol):
        if symbol not in self.latest_symbol_data:
            print(f"Warning: Symbol {symbol} not found in data")
            return None
        if not self.latest_symbol_data[symbol]:
            return None
        return self.symbol_data[symbol]['datetime'][len(self.latest_symbol_data[symbol]) - 1]

    def get_latest_bar_value(self, symbol, val_type):
        """Get the latest bar value for a specific symbol and value type"""
        try:
            # Normalize val_type to lowercase
            val_type = val_type.lower()
            
            # Map common column names
            column_mapping = {
                'close': 'close',
                'open': 'open', 
                'high': 'high',
                'low': 'low',
                'volume': 'volume',
                'price': 'close'  # Map 'price' to 'close'
            }
            
            # Get the actual column name
            actual_col = column_mapping.get(val_type, val_type)
            
            if symbol not in self.symbol_data or actual_col not in self.symbol_data[symbol]:
                print(f"Warning: Column '{actual_col}' not found for symbol '{symbol}'")
                return None
            
            data = self.symbol_data[symbol][actual_col]
            latest_data = self.latest_symbol_data[symbol]
            
            if not latest_data:
                return None
            
            # Get the last value
            idx = len(latest_data) - 1
            
            # Convert GPU array to CPU if needed
            if hasattr(data, 'get'):  # GPU array
                return float(cp.asnumpy(data[idx]))
            else:  # CPU array
                return float(data[idx])
                
        except Exception as e:
            print(f"Error getting latest bar value for {symbol}.{val_type}: {str(e)}")
            return None

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """Get the latest N bar values for a specific symbol and value type"""
        try:
            # Normalize val_type to lowercase for better compatibility
            val_type = val_type.lower()
            
            # Map common column names
            column_mapping = {
                'close': 'close',
                'open': 'open', 
                'high': 'high',
                'low': 'low',
                'volume': 'volume',
                'price': 'close'  # Map 'price' to 'close'
            }
            
            # Get the actual column name
            actual_col = column_mapping.get(val_type, val_type)
            
            if symbol not in self.symbol_data or actual_col not in self.symbol_data[symbol]:
                print(f"Warning: Column '{actual_col}' not found for symbol '{symbol}'")
                print(f"Available columns: {list(self.symbol_data.get(symbol, {}).keys())}")
                return []
            
            data = self.symbol_data[symbol][actual_col]
            latest_data = self.latest_symbol_data[symbol]
            
            if not latest_data:
                return []
            
            # Get the last N values
            start_idx = max(0, len(latest_data) - N)
            end_idx = len(latest_data)
            
            # Convert GPU array to CPU if needed
            if hasattr(data, 'get'):  # GPU array
                return cp.asnumpy(data[start_idx:end_idx]).tolist()
            else:  # CPU array
                return data[start_idx:end_idx].tolist()
                
        except Exception as e:
            print(f"Error getting latest bars values for {symbol}.{val_type}: {str(e)}")
            return []

    def update_bars(self):
        """GPU-accelerated bar updates with batch processing"""
        try:
            if self._current_index >= self._total_bars:
                self.continue_backtest = False
                return

            # Process in batches for better performance
            end_idx = min(self._current_index + self._batch_size, self._total_bars)
            
            # Debug output (much less frequent)
            if hasattr(self, '_update_count'):
                self._update_count += 1
            else:
                self._update_count = 1
                
            if self._update_count % 10000 == 0:
                print(f"DataHandler: Processing bars {self._current_index} to {end_idx} (total: {self._total_bars})")
            
            # Update all symbols in batch
            for symbol in self.symbol_list:
                if symbol not in self.symbol_data:
                    continue
                    
                # Get batch data from GPU
                batch_data = {
                    'datetime': self.symbol_data[symbol]['datetime'][self._current_index:end_idx],
                    **{key: self.symbol_data[symbol][key][self._current_index:end_idx]
                       for key in self.required_cols.keys()
                       if key in self.symbol_data[symbol]}
                }
                
                # Convert GPU arrays to CPU for event processing
                batch_data_cpu = {
                    'datetime': batch_data['datetime'],
                    **{key: cp.asnumpy(batch_data[key])
                       for key in self.required_cols.keys()
                       if key in batch_data}
                }
                
                self.latest_symbol_data[symbol].extend(
                    [{**{'datetime': dt}, **{k: v[i] for k, v in batch_data_cpu.items() if k != 'datetime'}}
                     for i, dt in enumerate(batch_data_cpu['datetime'])]
                )
            
            # Update current index
            self._current_index = end_idx
            
            # Create market event
            self.events.put(MarketEvent())
            
            if self._update_count % 10000 == 0:
                print(f"DataHandler: Created market event, latest data count: {len(self.latest_symbol_data[symbol])}")
            
        except Exception as e:
            print(f"Error in update_bars: {str(e)}")
            self.continue_backtest = False

    def get_batch_data(self, symbol, val_type, batch_size=None):
        """Optimized batch data retrieval"""
        if batch_size is None:
            batch_size = self._batch_size
            
        if not self.latest_symbol_data[symbol]:
            return None
            
        start_idx = max(0, len(self.latest_symbol_data[symbol]) - batch_size)
        end_idx = len(self.latest_symbol_data[symbol])
        
        # Use Numba-optimized batch processing
        return _process_batch_numba(
            self.symbol_data[symbol][val_type.lower()],
            start_idx,
            end_idx
        ) 
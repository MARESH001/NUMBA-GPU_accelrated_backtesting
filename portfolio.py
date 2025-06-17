from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from collections import defaultdict
import numpy as np
import cupy as cp
import pandas as pd
from datetime import datetime
from queue import Queue
from events import OrderEvent, FillEvent
import numba
from numba import cuda
import warnings
warnings.filterwarnings('ignore')

@cuda.jit
def calculate_returns_gpu(prices, returns):
    """GPU-accelerated returns calculation"""
    idx = cuda.grid(1)
    if idx > 0 and idx < prices.size:
        returns[idx] = (prices[idx] - prices[idx-1]) / prices[idx-1]

@cuda.jit
def calculate_equity_curve_gpu(returns, equity_curve):
    """GPU-accelerated equity curve calculation"""
    idx = cuda.grid(1)
    if idx < returns.size:
        if idx == 0:
            equity_curve[idx] = 1.0
        else:
            equity_curve[idx] = equity_curve[idx-1] * (1.0 + returns[idx])

@cuda.jit
def calculate_drawdowns_gpu(total_values, drawdown, cummax):
    """GPU-accelerated drawdown calculation"""
    idx = cuda.grid(1)
    if idx < total_values.size:
        if idx == 0:
            cummax[idx] = total_values[idx]
        else:
            cummax[idx] = max(cummax[idx-1], total_values[idx])
        drawdown[idx] = (total_values[idx] - cummax[idx]) / cummax[idx]

@dataclass
class Portfolio:
    data_handler: object
    events: Queue
    start_date: datetime
    initial_capital: float = 100000.0
    current_positions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    current_holdings: Dict[str, float] = field(default_factory=lambda: {
        "cash": 100000.0,
        "commission": 0.0,
        "total": 100000.0
    })
    all_positions: List[Dict[str, int]] = field(default_factory=list)
    all_holdings: List[Dict[str, Union[datetime, float]]] = field(default_factory=list)
    equity_curve: Optional[pd.DataFrame] = None
    _position_cache: Dict[str, int] = field(default_factory=dict)
    _market_value_cache: Dict[str, float] = field(default_factory=dict)
    _gpu_device: cp.cuda.Device = field(default_factory=lambda: cp.cuda.Device(0))
    _threads_per_block: int = 256

    def __post_init__(self):
        self.current_holdings["cash"] = self.initial_capital
        self.current_holdings["total"] = self.initial_capital
        self._position_cache = defaultdict(int)
        self._market_value_cache = {}

    def _calculate_returns_gpu(self, prices: np.ndarray) -> np.ndarray:
        """Calculate returns using GPU"""
        n = len(prices)
        returns = cp.zeros(n, dtype=cp.float64)
        
        # Transfer data to GPU
        prices_gpu = cp.asarray(prices)
        
        # Calculate grid dimensions
        blocks = (n + self._threads_per_block - 1) // self._threads_per_block
        
        # Launch CUDA kernel
        calculate_returns_gpu[blocks, self._threads_per_block](prices_gpu, returns)
        
        # Transfer results back to CPU
        return cp.asnumpy(returns)

    def _calculate_equity_curve_gpu(self, returns: np.ndarray) -> np.ndarray:
        """Calculate equity curve using GPU"""
        n = len(returns)
        equity_curve = cp.zeros(n, dtype=cp.float64)
        
        # Transfer data to GPU
        returns_gpu = cp.asarray(returns)
        
        # Calculate grid dimensions
        blocks = (n + self._threads_per_block - 1) // self._threads_per_block
        
        # Launch CUDA kernel
        calculate_equity_curve_gpu[blocks, self._threads_per_block](returns_gpu, equity_curve)
        
        # Transfer results back to CPU
        return cp.asnumpy(equity_curve)

    def _calculate_drawdowns_gpu(self, total_values: np.ndarray) -> tuple[np.ndarray, float]:
        """Calculate drawdowns using GPU"""
        n = len(total_values)
        drawdown = cp.zeros(n, dtype=cp.float64)
        cummax = cp.zeros(n, dtype=cp.float64)
        
        # Transfer data to GPU
        total_values_gpu = cp.asarray(total_values)
        
        # Calculate grid dimensions
        blocks = (n + self._threads_per_block - 1) // self._threads_per_block
        
        # Launch CUDA kernel
        calculate_drawdowns_gpu[blocks, self._threads_per_block](
            total_values_gpu, drawdown, cummax
        )
        
        # Transfer results back to CPU
        drawdown_cpu = cp.asnumpy(drawdown)
        max_dd = float(abs(cp.min(drawdown)))
        
        return drawdown_cpu, max_dd

    def update_timeindex(self, latest_datetime: datetime, market_data: Dict[str, float]) -> None:
        """Update portfolio holdings with GPU acceleration"""
        try:
            # Update position cache
            for symbol in market_data:
                self._position_cache[symbol] = self.current_positions.get(symbol, 0)
            
            # Convert to GPU arrays for faster calculation
            positions_array = cp.array(
                [self._position_cache[symbol] for symbol in market_data.keys()],
                dtype=cp.float64
            )
            prices_array = cp.array(list(market_data.values()), dtype=cp.float64)
            
            # Calculate market value on GPU
            market_value = float(cp.sum(positions_array * prices_array))
            
            # Update holdings
            new_holdings = {
                "datetime": latest_datetime,
                "cash": self.current_holdings["cash"],
                "commission": self.current_holdings["commission"],
                "total": self.current_holdings["cash"] + market_value
            }
            
            # Append to historical data
            self.all_positions.append(dict(self._position_cache))
            self.all_holdings.append(new_holdings)
            
            # Update current holdings
            self.current_holdings["total"] = new_holdings["total"]
            
        except Exception as e:
            print(f"Error in update_timeindex: {e}")

    def generate_order(self, signal) -> Optional[OrderEvent]:
        """Generate order from signal"""
        try:
            order = None
            symbol = signal.symbol
            direction = signal.signal_type
            quantity = int(100)  # Fixed quantity for simplicity

            if direction == 'LONG':
                order = OrderEvent(symbol, 'BUY', quantity, 'BUY')
            elif direction == 'EXIT':
                order = OrderEvent(symbol, 'SELL', quantity, 'SELL')

            return order
        except Exception as e:
            print(f"Error generating order: {e}")
            return None

    def update_fill(self, fill: FillEvent) -> None:
        """Update portfolio with fill information"""
        try:
            # Update positions
            if fill.direction == 'BUY':
                self.current_positions[fill.symbol] += fill.quantity
            else:  # SELL
                self.current_positions[fill.symbol] -= fill.quantity

            # Update cash and commission
            fill_cost = fill.fill_cost * fill.quantity
            self.current_holdings["cash"] -= fill_cost
            self.current_holdings["commission"] += fill.commission
            self.current_holdings["total"] -= fill.commission
            
            # Update caches
            self._position_cache[fill.symbol] = self.current_positions[fill.symbol]
            
        except Exception as e:
            print(f"Error updating fill: {e}")

    def create_equity_curve(self) -> pd.DataFrame:
        """Create equity curve DataFrame with GPU acceleration"""
        try:
            if not self.all_holdings:
                print("No holdings data available")
                return pd.DataFrame()
                
            df = pd.DataFrame(self.all_holdings)
            
            if 'datetime' not in df.columns:
                print("Error: 'datetime' column not found in holdings data")
                return pd.DataFrame()
                
            df.set_index("datetime", inplace=True)
            
            # Use GPU-accelerated functions for calculations
            total_values = df["total"].values
            returns = self._calculate_returns_gpu(total_values)
            equity_curve = self._calculate_equity_curve_gpu(returns)
            
            df["returns"] = returns
            df["equity_curve"] = equity_curve
            
            return df
        except Exception as e:
            print(f"Error creating equity curve: {str(e)}")
            return pd.DataFrame()

    def calculate_sharpe_ratio(self, returns: np.ndarray, periods: int = 252, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio using GPU acceleration"""
        try:
            if len(returns) < 2:
                return 0.0
                
            # Transfer data to GPU
            returns_gpu = cp.asarray(returns)
            
            # Remove NaN values
            returns_gpu = returns_gpu[~cp.isnan(returns_gpu)]
            if len(returns_gpu) < 2:
                return 0.0
                
            # Calculate on GPU
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            excess_returns = returns_gpu - daily_rf
            return_mean = float(cp.mean(excess_returns))
            return_std = float(cp.std(returns_gpu))
            
            if return_std == 0:
                return 0.0
                
            sharpe = np.sqrt(periods) * return_mean / return_std
            return sharpe if not np.isnan(sharpe) else 0.0
            
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def output_summary_stats(self) -> None:
        """Output performance metrics with GPU acceleration"""
        try:
            print("\nCalculating performance metrics...")
            
            if not self.all_holdings:
                print("No holdings data available")
                return
                
            # Convert holdings to DataFrame
            df = pd.DataFrame(self.all_holdings)
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            
            # Use GPU-accelerated functions for calculations
            total_values = df["total"].values
            returns = self._calculate_returns_gpu(total_values)
            equity_curve = self._calculate_equity_curve_gpu(returns)
            
            df["returns"] = returns
            df["equity_curve"] = equity_curve
            
            # Calculate metrics
            total_return = (total_values[-1] / total_values[0]) - 1.0
            if len(df) > 0 and total_return > -1:
                annualized_return = (1.0 + total_return) ** (252.0 / len(df)) - 1.0
            else:
                annualized_return = total_return
                
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
            
            # Calculate drawdowns using GPU
            drawdown, max_dd = self._calculate_drawdowns_gpu(total_values)
            
            # Print performance summary
            print("\nPerformance Summary:")
            print("=" * 50)
            print(f"Total Return         : {total_return*100:.2f}%")
            print(f"Annualized Return    : {annualized_return*100:.2f}%")
            print(f"Volatility           : {volatility*100:.2f}%")
            print(f"Sharpe Ratio         : {sharpe_ratio:.2f}")
            print(f"Maximum Drawdown     : {max_dd*100:.2f}%")
            print(f"Final Portfolio Value: ${total_values[-1]:,.2f}")
            print("=" * 50)
            
            # Plot equity curve
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        df.index = range(len(df))
                
                plt.plot(df.index, df["equity_curve"], label="Equity Curve")
                plt.title("Portfolio Equity Curve")
                plt.xlabel("Date" if isinstance(df.index, pd.DatetimeIndex) else "Trading Days")
                plt.ylabel("Portfolio Value")
                plt.grid(True)
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('equity_curve.png')
                plt.close()
                print("\nEquity curve saved as 'equity_curve.png'")
            except Exception as e:
                print(f"Error plotting equity curve: {e}")
            
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")

    def plot_equity_curve(self, save_path='equity_curve.png'):
        """Plot and save equity curve with GPU acceleration"""
        try:
            if not self.all_holdings:
                print("No holdings data available")
                return
                
            # Convert holdings to DataFrame
            df = pd.DataFrame(self.all_holdings)
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            
            # Use GPU-accelerated functions
            total_values = df["total"].values
            returns = self._calculate_returns_gpu(total_values)
            equity_curve = self._calculate_equity_curve_gpu(returns)
            
            df["returns"] = returns
            df["equity_curve"] = equity_curve
            
            # Plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    df.index = range(len(df))
            
            plt.plot(df.index, df["equity_curve"], label="Equity Curve")
            plt.title("Portfolio Equity Curve")
            plt.xlabel("Date" if isinstance(df.index, pd.DatetimeIndex) else "Trading Days")
            plt.ylabel("Portfolio Value")
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"\nEquity curve saved as '{save_path}'")
            
        except Exception as e:
            print(f"Error plotting equity curve: {e}")

if __name__ == "__main__":
    # Example usage:
    from data_handler import HistoricCSVDataHandler
    from events import Queue
    from datetime import datetime
    
    # Initialize components
    events = Queue()
    data_handler = HistoricCSVDataHandler(events, ".", ["MSFT"])
    portfolio = Portfolio(data_handler, events, datetime(2010, 1, 1))
    
    # Run backtest
    while data_handler.continue_backtest:
        data_handler.update_bars()
        # Process events if needed
    
    # Plot equity curve
    portfolio.plot_equity_curve()
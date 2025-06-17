from memory_profiler import profile
import pandas as pd
import numpy as np
from data_handler import HistoricCSVDataHandler
from strategy import NumbaStrategy
from portfolio import Portfolio
from backtest import Backtest
from events import MarketEvent
from execution import SimulatedExecutionHandler
from datetime import datetime
import time

@profile
def profile_data_handler():
    """Profile memory usage of DataHandler"""
    print("\nProfiling DataHandler...")
    events = MarketEvent()
    data_handler = HistoricCSVDataHandler(
        events=events,
        csv_dir=".",  # Current directory where CSV files are located
        symbol_list=["MSFT"]  # Using MSFT.csv as an example
    )
    return data_handler

@profile
def profile_strategy():
    """Profile memory usage of Strategy"""
    print("\nProfiling Strategy...")
    events = MarketEvent()
    data_handler = HistoricCSVDataHandler(
        events=events,
        csv_dir=".",
        symbol_list=["MSFT"]
    )
    strategy = NumbaStrategy(
        bars=data_handler,
        events=events,
        symbol="MSFT",
        window=20
    )
    # Generate some sample signals
    data = pd.read_csv("MSFT.csv")
    market_event = MarketEvent()
    strategy.calculate_signals(market_event)
    return strategy

@profile
def profile_portfolio():
    """Profile memory usage of Portfolio"""
    print("\nProfiling Portfolio...")
    events = MarketEvent()
    data_handler = HistoricCSVDataHandler(
        events=events,
        csv_dir=".",
        symbol_list=["MSFT"]
    )
    # Get the start date from the data
    data = pd.read_csv("MSFT.csv")
    start_date = pd.to_datetime(data['date'].iloc[0])
    
    portfolio = Portfolio(
        data_handler=data_handler,
        events=events,
        start_date=start_date,
        initial_capital=100000.0
    )
    # Create some sample positions
    market_data = {
        'MSFT': data['close'].iloc[0]
    }
    portfolio.update_timeindex(start_date, market_data)
    return portfolio

@profile
def profile_backtest():
    """Profile memory usage of Backtest"""
    print("\nProfiling Backtest...")
    events = MarketEvent()
    data_handler = HistoricCSVDataHandler(
        events=events,
        csv_dir=".",
        symbol_list=["MSFT"]
    )
    strategy = NumbaStrategy(
        bars=data_handler,
        events=events,
        symbol="MSFT",
        window=20
    )
    # Get the start date from the data
    data = pd.read_csv("MSFT.csv")
    start_date = pd.to_datetime(data['date'].iloc[0])
    
    portfolio = Portfolio(
        data_handler=data_handler,
        events=events,
        start_date=start_date,
        initial_capital=100000.0
    )
    
    execution_handler = SimulatedExecutionHandler(events, data_handler)
    
    backtest = Backtest(
        csv_dir=".",
        symbol_list=["MSFT"],
        initial_capital=100000.0,
        heartbeat=0.0,
        start_date=start_date,
        data_handler=data_handler,
        execution_handler=execution_handler,
        portfolio=portfolio,
        strategy=strategy
    )
    return backtest

def main():
    print("Starting memory profiling...")
    
    # Profile each component
    data_handler = profile_data_handler()
    strategy = profile_strategy()
    portfolio = profile_portfolio()
    backtest = profile_backtest()
    
    print("\nMemory profiling completed!")

if __name__ == "__main__":
    main() 
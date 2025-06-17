from abc import ABCMeta, abstractmethod
from events import FillEvent
import numpy as np

class ExecutionHandler(metaclass=ABCMeta):
    @abstractmethod
    def execute_order(self, event):
        raise NotImplementedError("Should implement execute_order()")

class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, events, data_handler, commission=0.001):
        self.events = events
        self.data_handler = data_handler
        self.commission = commission
        self.order_count = 0
        self.fill_count = 0
        
    def execute_order(self, event):
        if event.type == 'ORDER':
            self.order_count += 1
            
            # Get current market price
            fill_price = self.data_handler.get_latest_bar_value(event.symbol, "close")
            
            if fill_price is None:
                print(f"Warning: Could not get fill price for {event.symbol}")
                return
            
            # Calculate commission
            commission = self.calculate_commission(event.quantity, fill_price)
            
            # Create fill event
            fill_event = FillEvent(
                timeindex=self.data_handler.get_latest_bar_datetime(event.symbol),
                symbol=event.symbol,
                exchange='ARCA',
                quantity=event.quantity,
                direction=event.direction,
                fill_cost=fill_price,
                commission=commission
            )
            
            self.events.put(fill_event)
            self.fill_count += 1
            
            print(f"Order executed: {event.direction} {event.quantity} {event.symbol} at ${fill_price:.2f} (Commission: ${commission:.2f})")
            print(f"Total orders: {self.order_count}, Total fills: {self.fill_count}")

    def _create_fill_from_order(self, order):
        """Create a fill event from an order"""
        symbol = order.symbol
        quantity = order.quantity
        direction = order.direction
        price = self.data_handler.get_latest_bar_value(symbol, "close")
        
        if price is None:
            print(f"Warning: Could not get price for {symbol}")
            return None
            
        commission = self.calculate_commission(quantity, price)
        
        return FillEvent(
            self.data_handler.get_latest_bar_datetime(symbol),
            symbol,
            "ARCA",
            quantity,
            direction,
            price,
            commission
        )

    def calculate_commission(self, quantity, price):
        """Calculate commission based on quantity and price"""
        # Base commission: $1.30 minimum or $0.013 per share
        base_commission = max(1.30, 0.013 * quantity)
        
        # Add percentage-based commission
        percentage_commission = price * quantity * self.commission
        
        total_commission = base_commission + percentage_commission
        
        return total_commission 
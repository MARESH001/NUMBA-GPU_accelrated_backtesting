from dataclasses import dataclass, field
from typing import Optional, Union, ClassVar
from datetime import datetime

@dataclass
class Event:
    """Base class for all events"""
    EVENT_TYPE: ClassVar[str] = 'EVENT'
    
    @property
    def type(self) -> str:
        return self.EVENT_TYPE

@dataclass
class SignalEvent(Event):
    """Signal event for trading signals"""
    EVENT_TYPE: ClassVar[str] = 'SIGNAL'
    symbol: str
    datetime: Optional[datetime]
    signal_type: str
    strength: float

@dataclass
class OrderEvent(Event):
    """Order event for trade orders"""
    EVENT_TYPE: ClassVar[str] = 'ORDER'
    symbol: str
    order_type: str
    quantity: int
    direction: str

    def print_order(self) -> None:
        print(f"Order: Symbol={self.symbol}, Type={self.order_type}, Quantity={self.quantity}, Direction={self.direction}")

@dataclass
class FillEvent(Event):
    """Fill event for executed trades"""
    EVENT_TYPE: ClassVar[str] = 'FILL'
    timeindex: datetime
    symbol: str
    exchange: str
    quantity: int
    direction: str
    fill_cost: float
    commission: Optional[float] = None

    def __post_init__(self) -> None:
        if self.commission is None:
            self.commission = self.calculate_ib_commission()

    def calculate_ib_commission(self) -> float:
        """Calculate Interactive Brokers commission"""
        if self.quantity <= 500:
            return max(1.3, 0.013 * self.quantity)
        return max(1.3, 0.008 * self.quantity)

@dataclass
class MarketEvent(Event):
    """Market event for market data updates"""
    EVENT_TYPE: ClassVar[str] = 'MARKET' 
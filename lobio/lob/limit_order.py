from enum import IntEnum
from dataclasses import dataclass
PRICE_TICK: int = 2
AMOUNT_TICK: int = 3

class Side(IntEnum):
    """Class for LOB side: -1 if BUY, 1 if SELL."""

    BUY: int = -1
    SELL: int = 1

class TraderId(IntEnum):
    """Traders id: market, us (MM), or particular trader."""

    MARKET: int = 0
    MM: int = 1

class EventType(IntEnum):
    DIFF: int = 0
    LIMIT: int = 1
    MARKET: int = 2

class OrderType(IntEnum):
    LIMIT: int = 0
    MARKET: int = 1

@dataclass(slots=True)
class Order:
    base: int
    quote: int
    side: Side
    type: OrderType
    trader_id: TraderId

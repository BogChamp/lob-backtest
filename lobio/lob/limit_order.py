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

# class LimitOrder:
#     """Class for limit order imitation."""

#     def __init__(self, base: int, quote: int, side: int, trader_id: int):
#         """Creation of limit order.

#         Args:
#         ----
#             base (int): amount of based asset desired per quoted asset
#             quote (int): amount of quoted asset wanted to trade
#             side (int): 0 - buy order, 1 - sell order
#             trader_id (int): who set this limit order
#         """
#         # base = round(base, PRICE_TICK)
#         # quote = round(quote, AMOUNT_TICK)
#         assert quote > 0
#         assert base >= 0

#         self.base = base
#         self.quote = quote
#         self.side = side
#         self.trader_id = trader_id

#     def __repr__(self):
#         return f"LimitOrder({self.base}, {self.quote}, {self.side}, {self.trader_id})"
    
#     def __eq__(self, other): 
#         if not isinstance(other, LimitOrder):
#             return NotImplemented

#         return self.base == other.base and \
#             self.quote == other.quote and \
#             self.side == other.side and \
#             self.trader_id == other.trader_id

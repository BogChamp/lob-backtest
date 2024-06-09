import numpy as np
from lobio.lob.order_book import OrderBook
from lobio.lob.limit_order import Order, OrderType, PRICE_TICK, TraderId, Side
from lobio.strategies.base_model import Model
from typing import Tuple, Optional


class AvellanedaStoikov(Model):
    """Class containes Avellaneda Stoikov model implementation."""

    def __init__(
        self,
        T: int,
        t_start: int,
        q0: int = 0,
        k: float = 1.5,
        sigma: float = 2,
        gamma: float = 0.1,
        order_quote: int = 1,
    ):
        self.T = T
        self.t_start = t_start
        self.q0 = q0
        self.k = k
        self.sigma = sigma
        self.gamma = gamma
        self.order_quote = order_quote

        self.q = self.q0
    
    def reset(self):
        self.q = 0

    def get_indifference_price(self, mid_price: float, timestamp: int) -> float:
        if timestamp > self.T:
            raise Exception("Time for trading is over")

        r = mid_price - self.q * self.gamma * self.sigma**2 * (self.T - timestamp) / (
            self.T - self.t_start
        )
        return r

    def get_optimal_spread(self, timestamp: int) -> float:
        if timestamp > self.T:
            raise Exception("Time for trading is over")

        optimal_spread = self.gamma * self.sigma**2 * (self.T - timestamp) / (
            self.T - self.t_start
        ) + 2 / self.gamma * np.log1p(self.gamma / self.k)

        return optimal_spread

    def update_inventory(self, q: int):
        self.q = q

    def get_bid_ask_price(
        self, mid_price: float, timestamp: int
    ) -> Tuple[float, float]:
        r = self.get_indifference_price(mid_price, timestamp)
        optimal_spread = self.get_optimal_spread(timestamp)

        bid_price = r - optimal_spread / 2
        ask_price = r + optimal_spread / 2

        bid_price = round(bid_price)#round(bid_price, PRICE_TICK)
        ask_price = round(ask_price)#round(ask_price, PRICE_TICK)

        return bid_price, ask_price
    # MAKERS in TOP_N WINDOW
    def bid_ask_limit_orders( 
        self, lob_state: OrderBook, timestamp: int, q: Optional[int] = None
    ) -> Tuple[list[Order], list[Order]]:
        if q is not None:
            self.update_inventory(q)
        bid_price, ask_price = self.get_bid_ask_price(lob_state.mid_price(), timestamp)

        if bid_price < lob_state.ask_price() and bid_price <= lob_state.bids[-1].base:
            bid_orders = [Order(bid_price, self.order_quote, Side.BUY, OrderType.LIMIT, TraderId.MM)]
        else:
            bid_orders = []
        
        if ask_price > lob_state.bid_price() and ask_price <= lob_state.asks[-1].base:
            ask_orders = [Order(ask_price, self.order_quote, Side.SELL, OrderType.LIMIT, TraderId.MM)]
        else:
            ask_orders = []
        return bid_orders, ask_orders

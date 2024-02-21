import numpy as np
from lobio.lob.order_book import OrderBook, TraderId, Side
from lobio.lob.limit_order import LimitOrder, PRICE_TICK
from lobio.model.base_model import Model
from typing import Tuple, Optional


class AvellanedaStoikov(Model):
    """Class containes Avellaneda Stoikov model implementation."""

    def __init__(
        self,
        T: float,
        t_start: float,
        q0: float = 0,
        k: float = 1.5,
        sigma: float = 2,
        gamma: float = 0.1,
        q_max: float = 1,
    ):
        """Set AS model parameters.

        Args:
        ----
            T (float): end timestamp in nanoceonds
            t_start (float): start timestamp in nanoseconds
            q0 (float, optional): initial quote. Defaults to 0.
            k (float, optional): k parameter. Defaults to 1.5.
            sigma (float, optional): volatility parameter. Defaults to 2.
            gamma (float, optional): gamma or risk parameter. Defaults to 0.1.
            q_max (float, optional): limitation for quote. Defaults to 1.
        """
        self.T = T
        self.t_start = t_start
        self.q0 = q0
        self.k = k
        self.sigma = sigma
        self.gamma = gamma
        self.q_max = q_max

        self.q = self.q0
    
    def reset(self):
        """Setting quote to 0."""
        self.q = 0

    def get_indifference_price(self, mid_price: float, timestamp) -> float:
        """Calculate updated mid price.

        Args:
        ----
            mid_price (float): mide price of current LOB.
            timestamp (_type_): current timestamp in nanoseconds.

        Raises:
        ------
            Exception: if trading time over (if timestamp > T)

        Returns:
        -------
            float: indifference_price
        """
        if timestamp > self.T:
            raise Exception("Time for trading is over")

        r = mid_price - self.q * self.gamma * self.sigma**2 * (self.T - timestamp) / (
            self.T - self.t_start
        )
        return r

    def get_optimal_spread(self, timestamp: float) -> float:
        """Calculate optimal spread.

        Args:
        ----
            timestamp (float): current timestamp in nanoseconds.

        Raises:
        ------
            Exception: if trading time over (if timestamp > T)

        Returns:
        -------
            float: optimal spread
        """
        if timestamp > self.T:
            raise Exception("Time for trading is over")

        optimal_spread = self.gamma * self.sigma**2 * (self.T - timestamp) / (
            self.T - self.t_start
        ) + 2 / self.gamma * np.log1p(self.gamma / self.k)

        return optimal_spread

    def update_inventory(self, q: float):
        """Update information of traders quote.

        Args:
        ----
            q (float): current traders quote amount.
        """
        self.q = q

    def get_bid_ask_price(
        self, lob_state: OrderBook, timestamp: float
    ) -> Tuple[float, float]:
        """Calculate prices for bid and ask orders.

        Args:
        ----
            lob_state (OrderBook): current lob state.
            timestamp (float):current timestamp in nanoseconds.

        Returns:
        -------
            Tuple[float, float]: pair of bid and ask price
        """
        r = self.get_indifference_price(lob_state.mid_price(), timestamp)
        optimal_spread = self.get_optimal_spread(timestamp)

        bid_price = r - optimal_spread / 2
        ask_price = r + optimal_spread / 2

        bid_price = round(bid_price, PRICE_TICK)
        ask_price = round(ask_price, PRICE_TICK)

        return bid_price, ask_price

    def bid_ask_limit_orders(
        self, lob_state: OrderBook, timestamp: float, q: Optional[float] = None
    ) -> Tuple[list[LimitOrder], list[LimitOrder]]:
        """Creation of bid and ask orders lists.

        Args:
        ----
            lob_state (OrderBook): current lob state.
            timestamp (float): current timestamp in nanoseconds.
            q (Optional[float], optional): current traders quote amount. Defaults to None.

        Returns:
        -------
            Tuple[Sequence[LimitOrder], Sequence[LimitOrder]]: pair of bid orders and ask orders.
        """
        if q is not None:
            self.update_inventory(q)
        bid_price, ask_price = self.get_bid_ask_price(lob_state, timestamp)

        if bid_price < lob_state.ask_price():
            bid_orders = [LimitOrder(bid_price, self.q_max, Side.BUY, TraderId.MM)]
        else:
            bid_orders = [LimitOrder(lob_state.bid_price(), self.q_max, Side.BUY, TraderId.MM)]
        
        if ask_price > lob_state.bid_price():
            ask_orders = [LimitOrder(ask_price, self.q_max, Side.SELL, TraderId.MM)]
        else:
            ask_orders = [LimitOrder(lob_state.ask_price(), self.q_max, Side.SELL, TraderId.MM)]

        return bid_orders, ask_orders

from .limit_order import LimitOrder, AMOUNT_TICK
from collections import defaultdict
from typing import Tuple, Any

class PriceLevel:
    """Class for FIFO logic handling on price level."""

    def __init__(self, first_limit_order: LimitOrder):
        """Creation of price level.

        Args:
        ----
            first_limit_order (LimitOrder): first limit order 
                                            which create this price level
        """
        self.base = first_limit_order.base
        self.quote = first_limit_order.quote
        self.side = first_limit_order.side
        self.traders_order = [first_limit_order]

    def __repr__(self):
        return f"PriceLevel({self.base}, {self.quote}, {self.side})"

    def add_limit_order(self, limit_order: LimitOrder):  # trader_id: 0 - market, 1 - MM
        """Addition of limit order to queue.

        Args:
        ----
            limit_order (LimitOrder): limit order for adding to a queue.
        """
        assert limit_order.base == self.base
        assert limit_order.side == self.side

        self.quote += limit_order.quote
        self.quote = round(self.quote, AMOUNT_TICK)
        if len(self.traders_order):
            if limit_order.trader_id == self.traders_order[-1].trader_id:
                self.traders_order[-1].quote += limit_order.quote
                self.traders_order[-1].quote = round(
                    self.traders_order[-1].quote, AMOUNT_TICK
                )
            else:
                self.traders_order.append(limit_order)
        else:
            self.traders_order.append(limit_order)

    def execute_limit_order(self, quote: float) -> Tuple[float, defaultdict[int]]:
        """Remove part of price level due to exchange.

        Args:
        ----
            quote (float): how much quote to remove

        Returns:
        -------
            Tuple(float, defaultdict[Any, int]): remain quote after exchange and 
                dictionary with keys of traders ids and values as exchanged amount of quoted asset per trader id 
        """
        remain_amount = round(quote, AMOUNT_TICK)
        match_info = defaultdict(int)  # trader_id - amount_sold

        for i, limit_order in enumerate(self.traders_order):
            match_info[limit_order.trader_id] += min(limit_order.quote, remain_amount)
            self.quote -= min(limit_order.quote, remain_amount)

            if remain_amount < limit_order.quote:
                limit_order.quote -= remain_amount
                limit_order.quote = round(limit_order.quote, AMOUNT_TICK)
                self.traders_order = self.traders_order[i:]
                remain_amount = 0
                break
            else:
                remain_amount -= limit_order.quote

        remain_amount = round(remain_amount, AMOUNT_TICK)
        self.quote = round(self.quote, AMOUNT_TICK)
        if self.quote == 0:
            self.traders_order = []

        return remain_amount, match_info

    def change_liquidity(self, quote: float, trader_id: int):
        """Change of traders quote amount.

        Args:
        ----
            quote (float): quote to change. If value positive - add to price level, 
            otherwise remove liquidity from traders orders.
            trader_id (int): whose quote to change
        """
        if quote > 0:
            limit_order = LimitOrder(self.base, quote, self.side, trader_id)
            self.add_limit_order(limit_order)
        else:
            quote = abs(quote)
            for i, limit_order in enumerate(self.traders_order):
                if limit_order.trader_id != trader_id:
                    continue
                self.quote -= min(limit_order.quote, quote)

                if quote < limit_order.quote:
                    limit_order.quote -= quote
                    limit_order.quote = round(limit_order.quote, AMOUNT_TICK)
                    break
                else:
                    quote -= limit_order.quote
                    del self.traders_order[i]

        self.quote = round(self.quote, AMOUNT_TICK)
        if self.quote == 0:
            self.traders_order = []
from .limit_order import LimitOrder, AMOUNT_TICK
from collections import defaultdict
from typing import Tuple, Sequence
from enum import IntEnum

class Side(IntEnum):
    """Class for LOB side: 0 if BUY, 1 if SELL."""

    BUY = 0
    SELL = 1


class TraderId(IntEnum):
    """Traders id: market, us (MM), or particular trader."""

    MARKET: int = 0
    MM: int = 1

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
    
    def __eq__(self, other):
        if not isinstance(other, PriceLevel):
            return NotImplemented
        
        return self.base == other.base and \
            self.quote == other.quote and \
            self.side == other.side and \
            self.traders_order == other.traders_order

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

    def execute_limit_order(self, quote: float) -> Tuple[float, defaultdict[int, float]]:
        """Remove part of price level due to exchange.

        Args:
        ----
            quote (float): how much quote to remove

        Returns:
        -------
            Tuple(float, defaultdict[int, float]): remain quote after exchange and 
                dictionary with keys of traders ids and values as exchanged amount of quoted asset per trader id 
        """
        remain_amount = round(quote, AMOUNT_TICK)
        match_info = defaultdict(int)  # trader_id - amount_sold

        for i, limit_order in enumerate(self.traders_order):
            min_quote = min(limit_order.quote, remain_amount)
            match_info[limit_order.trader_id] += min_quote
            self.quote -= min_quote

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

    def change_liquidity(self, quote: float, trader_id: int): # СЛИЯНИЕ ОРДЕРОВ У ОДИНАКОВЫХ ИСТОЧНИКОВ ПРОДУМАТЬ !!!
        """Change of traders quote amount.

        Args:
        ----
            quote (float): quote to change. If value positive - add to price level, 
            otherwise remove liquidity from traders orders.
            trader_id (int): whose quote to change
        """
        if quote > 0:
            limit_order = LimitOrder(self.base, quote, self.side, trader_id)
            self.add_limit_order(limit_order) # THINK ABOUT AMOUNT BEFORE
        else:
            quote = abs(quote)
            skipped_orders = []
            not_trader_amount = 0.0

            for i, limit_order in enumerate(self.traders_order): ### ORDER Implies test cases
                if limit_order.trader_id != trader_id:
                    skipped_orders.append(limit_order)
                    not_trader_amount += limit_order.quote
                    continue
                self.quote -= min(limit_order.quote, quote)

                if quote < limit_order.quote:
                    limit_order.quote -= quote
                    limit_order.quote = round(limit_order.quote, AMOUNT_TICK)
                    break
                else:
                    quote -= limit_order.quote

            not_trader_amount = round(not_trader_amount, AMOUNT_TICK)
            self.quote = round(self.quote, AMOUNT_TICK)

            if self.quote == not_trader_amount:
                self.traders_order = skipped_orders
            else:
                if i:
                    self.traders_order = skipped_orders + self.traders_order[i:]
        
        self.quote = round(self.quote, AMOUNT_TICK)
        if self.quote == 0:
            self.traders_order = []
    
    def change_liquidity_2(self, quote: float, trader_id: int):
        if quote > 0:
            limit_order = LimitOrder(self.base, quote, self.side, trader_id)
            self.add_limit_order(limit_order)
        else:
            quote = abs(quote)
            skipped_orders = []
            not_trader_amount = 0.0

            for i, limit_order in enumerate(self.traders_order[::-1]):
                if limit_order.trader_id != trader_id:
                    skipped_orders.append(limit_order)
                    not_trader_amount += limit_order.quote
                    continue
                self.quote -= min(limit_order.quote, quote)

                if quote < limit_order.quote:
                    limit_order.quote -= quote
                    limit_order.quote = round(limit_order.quote, AMOUNT_TICK)
                    break
                else:
                    quote -= limit_order.quote

            not_trader_amount = round(not_trader_amount, AMOUNT_TICK)
            self.quote = round(self.quote, AMOUNT_TICK)

            if self.quote == not_trader_amount:
                self.traders_order = skipped_orders[::-1]
            else:
                if i:
                    self.traders_order = self.traders_order[:-i] + skipped_orders[::-1]
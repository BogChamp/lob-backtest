from .limit_order import Order, AMOUNT_TICK, Side, TraderId, OrderType
from typing import Tuple, Sequence, Optional

class PriceLevel:
    """Class for FIFO logic handling on price level."""

    def __init__(self, first_limit_order: Order):
        """Creation of price level.

        Args:
        ----
            first_limit_order (LimitOrder): first limit order 
                                            which create this price level
        """
        self.base = first_limit_order.base
        self.quote = first_limit_order.quote
        self.history_quote = first_limit_order.quote if first_limit_order.trader_id == TraderId.MARKET else 0
        self.side = first_limit_order.side
        self.traders_order = [first_limit_order]

    def __repr__(self):
        return f"PriceLevel({self.base}, {self.quote}, {self.side})"
    
    def __eq__(self, other):
        if not isinstance(other, PriceLevel):
            return NotImplemented
        
        return self.base == other.base and \
            self.quote == other.quote and \
            self.history_quote == other.history_quote and \
            self.side == other.side and \
            self.traders_order == other.traders_order

    def add_limit_order(self, limit_order: Order):  # trader_id: 0 - market, 1 - MM
        """Addition of limit order to queue.

        Args:
        ----
            limit_order (LimitOrder): limit order for adding to a queue.
        """
        #assert limit_order.base == self.base and limit_order.side == self.side and limit_order.quote > 0

        self.quote += limit_order.quote
        if limit_order.trader_id == TraderId.MARKET:
            self.history_quote += limit_order.quote

        if limit_order.trader_id == self.traders_order[-1].trader_id:
            self.traders_order[-1].quote += limit_order.quote
        else:
            self.traders_order.append(limit_order)

    def execute_market_order(self, quote: int) -> Tuple[int, int]:#Tuple[int, dict[TraderId, int]]: # NO CHECK FOR SELF EXEC
        """Remove part of price level due to exchange.

        Args:
        ----
            quote (int): how much quote to remove

        Returns:
        -------
            Tuple(float, defaultdict[int, float]): remain quote after exchange and 
                dictionary with keys of traders ids and values as exchanged amount of quoted asset per trader id 
        """
        #assert quote > 0
        remain_amount = quote
        match_info = {TraderId.MARKET: 0, TraderId.MM: 0}  # trader_id - amount_sold

        for i, limit_order in enumerate(self.traders_order):
            min_quote = min(limit_order.quote, remain_amount) # trader id is also ind for quote
            match_info[limit_order.trader_id] -= self.side * min_quote # correct calculating of asset dynamic
            self.quote -= min_quote
            if limit_order.trader_id == TraderId.MARKET:
                self.history_quote -= min_quote

            if remain_amount < limit_order.quote:
                limit_order.quote -= remain_amount
                del self.traders_order[:i]
                remain_amount = 0
                break
            else:
                remain_amount -= limit_order.quote
        #assert self.quote >= 0
        return remain_amount, match_info[TraderId.MM]

    def change_historical_liquidity(self, quote: int):
        if quote > 0: # THINK ABOUT AMOUNT BEFORE
            self.quote += quote
            self.history_quote += quote

            if self.traders_order[-1].trader_id == TraderId.MARKET:
                self.traders_order[-1].quote += quote
            else:
                self.traders_order.append(Order(self.base, quote, self.side, OrderType.LIMIT, TraderId.MARKET)) 
        elif quote < 0:
            #assert quote < 0
            quote = abs(quote)
            skipped_orders = []
            
            for i, limit_order in enumerate(self.traders_order):
                if limit_order.trader_id != TraderId.MARKET:
                    skipped_orders.append(limit_order)
                    continue

                if quote < limit_order.quote:
                    limit_order.quote -= quote
                    self.quote -= quote
                    self.history_quote -= quote
                    break
                else:
                    quote -= limit_order.quote
                    self.quote -= limit_order.quote
                    self.history_quote -= limit_order.quote

            if self.history_quote == 0:
                self.traders_order = skipped_orders
            elif i:
                self.traders_order = skipped_orders + self.traders_order[i:]
    
    def change_liquidity_2(self, quote: int, trader_id: int):
        if quote > 0:
            limit_order = Order(self.base, quote, self.side, trader_id)
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
                    #limit_order.quote = round(limit_order.quote, AMOUNT_TICK)
                    break
                else:
                    quote -= limit_order.quote

            #not_trader_amount = round(not_trader_amount, AMOUNT_TICK)
            #self.quote = round(self.quote, AMOUNT_TICK)

            if self.quote == not_trader_amount:
                self.traders_order = skipped_orders[::-1]
            elif i:
                self.traders_order = self.traders_order[:-i] + skipped_orders[::-1]

        if self.quote == 0:
            self.traders_order = []


class PriceLevelSimple:
    def __init__(self, base:int, quote: int) -> None:
        self.amount = [quote, 0]
        self.base = base
        self.my_order_id = None
    
    def __repr__(self) -> str:
        return f'PriceLevel{self.base, self.amount}'

    def add_liquidity(self, quote: int):
        if self.amount[1] != 0: # if we are in price level
            self.amount[1] += quote
        else:
            self.amount[0] += quote
    
    def total_amount(self) -> int:
        return self.amount[0] + self.amount[1]
    
    def execute_market_order(self, quote: int) -> Tuple[int, int|None]: # always keep in ind 1 amount with us. After we executed, move liquidity to ind 0.
        if quote <= self.amount[0]:
            self.amount[0] -= quote
            quote = 0
            me_executed = None
        else:
            quote -= self.amount[0]
            if quote <= self.amount[1]:
                self.amount[1] -= quote
                quote = 0
                self.amount = [self.amount[1], 0]
            else:
                quote -= self.amount[1]
                self.amount = [0, 0]
            me_executed = self.my_order_id
            self.my_order_id = None

        return quote, me_executed

    def change_historical_liquidity_opt(self, quote: int):
        if quote > 0:
            self.add_liquidity(quote)
        elif quote < 0: # from start to end
            quote = abs(quote)
            if quote <= self.amount[0]:
                self.amount[0] -= quote
            else:
                quote -= self.amount[0]
                self.amount[0] = 0
                self.amount[1] -= quote
    
    def change_historical_liquidity(self, quote: int):
        if quote > 0:
            self.add_liquidity(quote)
        elif quote < 0: # from start to end
            quote = abs(quote)
            if quote <= self.amount[1] - 1:
                self.amount[1] -= quote
            else:
                quote -= self.amount[1] - 1
                self.amount[1] = 1
                self.amount[0] -= quote

    def place_my_order(self, ratio_after_me: float, order_id: int):
        quote_after_me = int(ratio_after_me * self.total_amount())
        self.amount[0] = self.total_amount() - quote_after_me - 1
        self.amount[1] = quote_after_me + 1
        self.my_order_id = order_id

    def queue_dynamic(self, ratio_after_me: float):
        quote_after_me = int(ratio_after_me * self.amount[0])
        self.amount[0] -= quote_after_me
        self.amount[1] += quote_after_me

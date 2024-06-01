from typing import Any, Sequence, Tuple, Self
from bisect import bisect_left

from .limit_order import Order, PRICE_TICK, AMOUNT_TICK, Side, TraderId, OrderType
from .price_level import PriceLevel, PriceLevelSimple

PRETTY_INDENT_OB: int = 36
TOP_N: int = 20

class OrderBook:
    """Order Book Simulator."""

    def __init__(
        self, bids: Sequence[PriceLevel] = None, asks: Sequence[PriceLevel] = None,
        top_n: int = TOP_N
    ):
        """Creation of order book with bids and asks.

        Args:
        ----
            bids (Sequence[PriceLevel], optional): all bids in ob. Defaults to None.
            asks (Sequence[PriceLevel], optional): all asks in ob. Defaults to None.
        """
        if bids is None:
            bids = []

        if asks is None:
            asks = []

        self.bids = sorted(bids, reverse=True, key=lambda x: x.base)
        self.asks = sorted(asks, key=lambda x: x.base)
        del self.bids[top_n:]
        del self.asks[top_n:]
        self.top_n = top_n

    def __repr__(self):
        ob_repr = ""

        min_len = min(len(self.bids), len(self.asks))
        for i in range(min_len):
            bid_str = repr(self.bids[i])
            ask_str = repr(self.asks[i])
            ob_repr += (
                bid_str
                + (PRETTY_INDENT_OB - len(bid_str)) * " "
                + ask_str
                + "\n"
            )

        if len(self.bids) > min_len:
            remain_price_levels = self.bids[min_len:]
            indent = 0
        else:
            remain_price_levels = self.asks[min_len:]
            indent = PRETTY_INDENT_OB

        for p_l in remain_price_levels:
            ob_repr += indent * " " + repr(p_l) + "\n"

        return ob_repr
    
    def __eq__(self, other):
        if not isinstance(other, OrderBook):
            return NotImplemented
        
        return self.bids == other.bids and self.asks == other.asks

    def get_state(self):
        return self.bids, self.asks

    def get_bids(self):
        return self.bids

    def get_asks(self):
        return self.asks

    def bid_price(self):
        return self.bids[0].base

    def ask_price(self):
        return self.asks[0].base

    def mid_price(self) -> float:
        return (self.bids[0].base + self.asks[0].base) / 2

    def bid_ask_spread(self):
        return self.asks[0].base - self.bids[0].base

    def market_depth(self):
        return self.asks[-1].base - self.bids[-1].base

    def get_top_n(self, n: int = 10) -> Tuple[list[PriceLevel], list[PriceLevel]]:
        """Returns top n bids and asks

        Args:
            n (int, optional): number of levels to return. Defaults to 10.

        Returns:
            Tuple[list[PriceLevel], list[PriceLevel]]: tuple of bids and asks lists with n top levels.
        """
        return self.bids[:n], self.asks[:n]

    def set_limit_order(self, limit_order: Order):
        if limit_order.side == Side.BUY:
            self.set_bid_limit_order(limit_order)
        else:
            self.set_ask_limit_order(limit_order)
    
    def set_bid_limit_order(self, limit_order: Order):
        index = bisect_left(self.bids, -limit_order.base, key=lambda x: -x.base)
        if index == len(self.bids):
            self.bids.append(PriceLevel(limit_order))
        elif self.bids[index].base == limit_order.base:
            self.bids[index].add_limit_order(limit_order)
        else:
            self.bids.insert(index, PriceLevel(limit_order))
    
    def set_ask_limit_order(self, limit_order: Order):
        index = bisect_left(self.asks, limit_order.base, key=lambda x: x.base)
        if index == len(self.asks):
            self.asks.append(PriceLevel(limit_order))
        elif self.asks[index].base == limit_order.base:
            self.asks[index].add_limit_order(limit_order)
        else:
            self.asks.insert(index, PriceLevel(limit_order))

    def set_market_order(self, market_order: Order) -> Tuple[int, int]:
        if market_order.side == Side.BUY:
            my_match_info = self.set_buy_market_order(market_order)
        else:
            my_match_info = self.set_sell_market_order(market_order)
        
        return my_match_info
    
    def set_buy_market_order(self, market_order: Order) -> Tuple[int, int]:
        # matches_info = {TraderId.MM: [0, 0], TraderId.MARKET: [0, 0]}
        matches_info = [0, 0]
        
        if not len(self.asks):
            return matches_info
        
        remain_amount = market_order.quote
        for i, price_level in enumerate(self.asks):
            this_price = price_level.base
            remain_amount, cur_match_info = price_level.execute_market_order(remain_amount)
            # for k, v in cur_match_info.items():
            #     trader_info = matches_info[k]
            matches_info[0] += cur_match_info
            matches_info[1] -= cur_match_info * this_price

            if remain_amount == 0:
                break

        if self.asks[i].quote == 0:
            i += 1
        del self.asks[:i] #self.asks = self.asks[i:]

        # if market_order.trader_id == TraderId.MM: # I'M NOT MAKER
        #     matches_info[TraderId.MM][0] -= matches_info[TraderId.MARKET][0]
        #     matches_info[TraderId.MM][1] -= matches_info[TraderId.MARKET][1]
        return matches_info #matches_info[TraderId.MM]
    
    def set_sell_market_order(self, market_order: Order) -> Tuple[int, int]:
        #matches_info = {TraderId.MM: [0, 0], TraderId.MARKET: [0, 0]}
        matches_info = [0, 0]

        if not len(self.bids):
            return matches_info

        remain_amount = market_order.quote
        for i, price_level in enumerate(self.bids):
            this_price = price_level.base
            remain_amount, cur_match_info = price_level.execute_market_order(remain_amount)
            # for k, v in cur_match_info.items():
            #     trader_info = matches_info[k]
            matches_info[0] += cur_match_info
            matches_info[1] -= cur_match_info * this_price

            if remain_amount == 0:
                break

        if self.bids[i].quote == 0:
            i += 1
        del self.bids[:i] #self.bids = self.bids[i:]

        # if market_order.trader_id == TraderId.MM:  # I'M NOT MAKER
        #     matches_info[TraderId.MM][0] -= matches_info[TraderId.MARKET][0]
        #     matches_info[TraderId.MM][1] -= matches_info[TraderId.MARKET][1]
        return matches_info #matches_info[TraderId.MM]

    @classmethod
    def create_lob_init(cls, lob_state: dict) -> Self:
        """Creation of lob from dictionary data.

        Args:
        ----
            lob_state (dict): snapshot of lob.

        Returns:
        -------
            OrderBook: new OrderBook instance.
        """
        bids_raw = lob_state["bids"]
        asks_raw = lob_state["asks"]

        bids = []
        asks = []
        for bid_raw in bids_raw:
            bid = PriceLevel(
                Order(bid_raw[0], bid_raw[1], Side.BUY, OrderType.LIMIT, TraderId.MARKET)
            )
            bids.append(bid)

        for ask_raw in asks_raw:
            ask = PriceLevel(
                Order(ask_raw[0], ask_raw[1], Side.SELL, OrderType.LIMIT, TraderId.MARKET)
            )
            asks.append(ask)

        return cls(bids, asks)

    def update_bids(self, updates: Sequence[Tuple[int, int]]):
        for update in updates:
            price = update[0]
            amount = update[1]
            index = bisect_left(
                self.bids,  -price, key=lambda x: -x.base
            )
            if index == len(self.bids):
                if amount > 0:
                    self.bids.append(
                        PriceLevel(Order(price, amount, Side.BUY, OrderType.LIMIT, TraderId.MARKET)) # mb diffs to order
                    )
            elif self.bids[index].base == price:
                amount_change = amount - self.bids[index].history_quote
                self.bids[index].change_historical_liquidity(amount_change)
                if self.bids[index].quote == 0:
                    del self.bids[index]
            else:
                if amount > 0:
                    self.bids.insert(
                        index,
                        PriceLevel(Order(price, amount, Side.BUY, OrderType.LIMIT, TraderId.MARKET)),
                    )
        del self.bids[self.top_n:]

    def update_asks(self, updates: Sequence[Tuple[int, int]]):
        for update in updates:
            price = update[0]
            amount = update[1]
            index = bisect_left(
                self.asks,  price, key=lambda x: x.base
            )
            if index == len(self.asks):
                if amount > 0:
                    self.asks.append(
                        PriceLevel(Order(price, amount, Side.SELL, OrderType.LIMIT, TraderId.MARKET))
                    )
            elif self.asks[index].base == price:
                amount_change = amount - self.asks[index].history_quote
                self.asks[index].change_historical_liquidity(amount_change)
                if self.asks[index].quote == 0:
                    del self.asks[index]
            else:
                if amount > 0:
                    self.asks.insert(
                        index,
                        PriceLevel(Order(price, amount, Side.SELL, OrderType.LIMIT, TraderId.MARKET)),
                    )
        del self.asks[self.top_n:]

    def apply_historical_update(self, updates: Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]):
        """Perform changes on order book according to historical dynamic movements of LOB.

        Args:
        ----
            updates (Tuple[float, Sequence, Sequence]): timestamp, list of updates for bids as pairs base-quote,
            list of updates for asks as pair base-quote
        """
        bids_update = updates[1]
        asks_update = updates[2]

        self.update_bids(bids_update)
        self.update_asks(asks_update)

    def remove_bid_ask_intersection(self) -> Tuple[int, int]:
        if not len(self.bids) or not len(self.asks):
            return [0, 0]

        my_data = [0, 0]
        while len(self.bids) and len(self.asks) and self.bids[0].base >= self.asks[0].base:
            if self.bids[0].quote > self.asks[0].quote:
                limit_orders = self.asks[0].traders_order
                del self.asks[0]
            else:
                limit_orders = self.bids[0].traders_order
                del self.bids[0]

            for limit_order in limit_orders:
                cur_match_info = self.set_market_order(limit_order)
                my_data[0] += cur_match_info[0]
                my_data[1] += cur_match_info[0]
        return my_data


class OrderBookSimple:
    def __init__(self, 
                 bids: Sequence[Tuple[int, int]] = None, 
                 asks: Sequence[Tuple[int, int]] = None):
        if bids is None:
            bids = []

        if asks is None:
            asks = []

        self.bids = sorted(bids, reverse=True, key=lambda x: x[0])
        self.asks = sorted(asks, key=lambda x: x[0])
    
    def __eq__(self, other):
        if not isinstance(other, OrderBookSimple):
            return NotImplemented
        
        return self.bids == other.bids and self.asks == other.asks

    @classmethod
    def create_lob_init(cls, lob_state: dict) -> Self:
        bids_raw = lob_state["bids"]
        asks_raw = lob_state["asks"]

        bids = []
        asks = []
        for bid_raw in bids_raw:
            bids.append([bid_raw[0], bid_raw[1]])

        for ask_raw in asks_raw:
            asks.append([ask_raw[0], ask_raw[1]])

        return cls(bids, asks)

    def apply_historical_update(self, diff: Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]):
        bids_update = diff[1]
        asks_update = diff[2]

        self.update_bids(bids_update)
        self.update_asks(asks_update)
    
    def update_bids(self, updates: Sequence[Tuple[int, int]]):
        for update in updates:
            price = update[0]
            amount = update[1]
            index = bisect_left(
                self.bids, -price, key=lambda x: -x[0]
            )
            if index == len(self.bids):
                if amount > 0:
                    self.bids.append([price, amount])
            elif self.bids[index][0] == price:
                if amount == 0:
                    del self.bids[index]
                else:
                    self.bids[index][1] = amount
            else:
                if amount > 0:
                    self.bids.insert(index, [price, amount])

    def update_asks(self, updates: Sequence[Tuple[int, int]]):
        for update in updates:
            price = update[0]
            amount = update[1]
            index = bisect_left(
                self.asks,  price, key=lambda x: x[0]
            )
            if index == len(self.asks):
                if amount > 0:
                    self.asks.append([price, amount])
            elif self.asks[index][0] == price:
                if amount == 0:
                    del self.asks[index]
                else:
                    self.asks[index][1] = amount
            else:
                if amount > 0:
                    self.asks.insert(index, [price, amount])

    def track_diff(self, diff: Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]) -> \
                                    Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]:
        bids_update = diff[1]
        asks_update = diff[2]

        bids_diff_new = self.track_bids(bids_update)
        asks_diff_new = self.track_asks(asks_update)

        return (diff[0], bids_diff_new, asks_diff_new)
    
    def track_bids(self, updates: Sequence[Tuple[int, int]]) -> Sequence[Tuple[int, int]]:
        diff_new = []
        for update in updates:
            price = update[0]
            amount = update[1]
            index = bisect_left(
                self.bids,  -price, key=lambda x: -x[0]
            )
            if index == len(self.bids):
                if amount > 0:
                    self.bids.append([price, amount])
                    diff_new.append([price, amount])
            elif self.bids[index][0] == price:
                if amount == 0:
                    del self.bids[index]
                else:
                    # amount_change = amount - self.bids[index][1]
                    self.bids[index][1] = amount
                diff_new.append([price, amount])
            else:
                if amount > 0:
                    self.bids.insert(index, [price, amount])
                    diff_new.append([price, amount])
        return diff_new

    def track_asks(self, updates: Sequence[Tuple[int, int]]) -> Sequence[Tuple[int, int]]:
        diff_new = []
        for update in updates:
            price = update[0]
            amount = update[1]
            index = bisect_left(
                self.asks,  price, key=lambda x: x[0]
            )
            if index == len(self.asks):
                if amount > 0:
                    self.asks.append([price, amount])
                    diff_new.append([price, amount])
            elif self.asks[index][0] == price:
                if amount == 0:
                    del self.asks[index]
                else:
                    self.asks[index][1] = amount
                diff_new.append([price, amount])
            else:
                if amount > 0:
                    self.asks.insert(index, [price, amount])
                    diff_new.append([price, amount])
        return diff_new

    def set_limit_order(self, limit_order: Tuple[int, int, int]):
        if limit_order[2] == Side.BUY:
            self.set_bid_limit_order(limit_order)
        else:
            self.set_ask_limit_order(limit_order)
    
    def set_bid_limit_order(self, limit_order: Tuple[int, int, int]):
        index = bisect_left(self.bids, -limit_order[0], key=lambda x: -x[0])
        if index == len(self.bids):
            self.bids.append([limit_order[0], limit_order[1]])
        elif self.bids[index][0] == limit_order[0]:
            self.bids[index][1] += limit_order[1]
        else:
            self.bids.insert(index, [limit_order[0], limit_order[1]])
    
    def set_ask_limit_order(self, limit_order: Tuple[int, int, int]):
        index = bisect_left(self.asks, limit_order[0], key=lambda x: x[0])
        if index == len(self.asks):
            self.asks.append([limit_order[0], limit_order[1]])
        elif self.asks[index][0] == limit_order[0]:
            self.asks[index][1] += limit_order[1]
        else:
            self.asks.insert(index, [limit_order[0], limit_order[1]])

    def set_market_order(self, market_order: Tuple[int, int]):
        if market_order[1] == Side.BUY:
            self.set_buy_market_order(market_order)
        else:
            self.set_sell_market_order(market_order)
    
    def set_buy_market_order(self, market_order: Tuple[int, int]):
        i = 0
        while i < len(self.asks) and self.asks[i][1] <= market_order[0]:
            market_order[0] -= self.asks[i][1]
            i += 1

        if i == len(self.asks):
            self.asks = []
        else:
            self.asks[i][1] -= market_order[0]
            del self.asks[:i]
    
    def set_sell_market_order(self, market_order: Tuple[int, int]):
        i = 0
        while i < len(self.bids) and self.bids[i][1] <= market_order[0]:
            market_order[0] -= self.bids[i][1]
            i += 1

        if i == len(self.bids):
            self.bids = []
        else:
            self.bids[i][1] -= market_order[0]
            del self.bids[:i]


class OrderBookTrunc(OrderBookSimple):
    def __init__(self, 
                 bids: Sequence[Tuple[int, int]] = None, 
                 asks: Sequence[Tuple[int, int]] = None,
                 top_n: int = TOP_N):
        super().__init__(bids, asks)
        del self.bids[top_n:]
        del self.asks[top_n:]
        self.top_n = top_n
    
    def update_bids(self, updates: Sequence[Tuple[int, int]]):
        super().update_bids(updates)
        del self.bids[self.top_n:]
    
    def update_asks(self, updates: Sequence[Tuple[int, int]]):
        super().update_asks(updates)
        del self.asks[self.top_n:]


class OrderBookSimple2:
    def __init__(self, 
                 bids: Sequence[PriceLevelSimple] = None, 
                 asks: Sequence[PriceLevelSimple] = None,
                 top_n: int = TOP_N):
        if bids is None:
            bids = []

        if asks is None:
            asks = []

        self.bids = sorted(bids, reverse=True, key=lambda x: x.base)
        self.asks = sorted(asks, key=lambda x: x.base)
        self.top_n = top_n
        del self.bids[self.top_n:]
        del self.asks[self.top_n:]
    
    @classmethod
    def create_lob_init(cls, lob_state: dict) -> Self:
        bids_raw = lob_state["bids"]
        asks_raw = lob_state["asks"]

        bids = []
        asks = []
        for bid_raw in bids_raw:
            bids.append(PriceLevelSimple(bid_raw[0], bid_raw[1]))

        for ask_raw in asks_raw:
            asks.append(PriceLevelSimple(ask_raw[0], ask_raw[1]))

        return cls(bids, asks)

    def apply_historical_update(self, diff: Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]) -> list[int]:
        bids_update = diff[1]
        asks_update = diff[2]

        my_orders_removed1 = self.update_bids(bids_update)
        my_orders_removed2 = self.update_asks(asks_update)

        return my_orders_removed1 + my_orders_removed2

    def update_bids(self, updates: Sequence[Tuple[int, int]]) -> list[int]:
        my_orders_removed = []

        for update in updates:
            price = update[0]
            amount = update[1]
            index = bisect_left(
                self.bids, -price, key=lambda x: -x.base
            )
            if index == len(self.bids):
                if amount > 0:
                    self.bids.append(PriceLevelSimple(price, amount))
            elif self.bids[index].base == price:
                if amount == 0:
                    if self.bids[index].my_order_id != None:
                        my_orders_removed.append(self.bids[index].my_order_id)
                    del self.bids[index]
                else:
                    amount_change = amount - self.bids[index].total_amount()
                    self.bids[index].change_historical_liquidity(amount_change) # change will never clear price level
            else:
                if amount > 0:
                    self.bids.insert(index, PriceLevelSimple(price, amount))

        for bid in self.bids[self.top_n:]:
            if bid.my_order_id != None:
                my_orders_removed.append(bid.my_order_id)
        del self.bids[self.top_n:]

        return my_orders_removed

    def update_asks(self, updates: Sequence[Tuple[int, int]]) -> list[int]:
        my_orders_removed = []

        for update in updates:
            price = update[0]
            amount = update[1]
            index = bisect_left(
                self.asks,  price, key=lambda x: x.base
            )
            if index == len(self.asks):
                if amount > 0:
                    self.asks.append(PriceLevelSimple(price, amount))
            elif self.asks[index].base == price:
                if amount == 0:
                    if self.asks[index].my_order_id != None:
                        my_orders_removed.append(self.asks[index].my_order_id)
                    del self.asks[index]
                else:
                    amount_change = amount - self.asks[index].total_amount()
                    self.asks[index].change_historical_liquidity(amount_change)
            else:
                if amount > 0:
                    self.asks.insert(index, PriceLevelSimple(price, amount))

        for ask in self.asks[self.top_n:]:
            if ask.my_order_id != None:
                my_orders_removed.append(ask.my_order_id)
        del self.asks[self.top_n:]

        return my_orders_removed
    
    def set_limit_order(self, limit_order: Tuple[int, int, int]):
        if limit_order[2] == Side.BUY:
            self.set_bid_limit_order(limit_order)
        else:
            self.set_ask_limit_order(limit_order)
    
    def set_bid_limit_order(self, limit_order: Tuple[int, int, int]):
        index = bisect_left(self.bids, -limit_order[0], key=lambda x: -x.base)
        if index == len(self.bids):
            self.bids.append(PriceLevelSimple(limit_order[0], limit_order[1]))
        elif self.bids[index].base == limit_order[0]:
            self.bids[index].add_liquidity(limit_order[1])
        else:
            self.bids.insert(index, PriceLevelSimple(limit_order[0], limit_order[1]))
    
    def set_ask_limit_order(self, limit_order: Tuple[int, int, int]):
        index = bisect_left(self.asks, limit_order[0], key=lambda x: x.base)
        if index == len(self.asks):
            self.asks.append(PriceLevelSimple(limit_order[0], limit_order[1]))
        elif self.asks[index].base == limit_order[0]:
            self.asks[index].add_liquidity(limit_order[1])
        else:
            self.asks.insert(index, PriceLevelSimple(limit_order[0], limit_order[1]))

    def set_market_order(self, market_order: Tuple[int, int]) -> list[int]:
        if market_order[1] == Side.BUY:
            my_orders_eaten = self.set_buy_market_order(market_order)
        else:
            my_orders_eaten = self.set_sell_market_order(market_order)
        
        return my_orders_eaten

    def set_buy_market_order(self, market_order: Tuple[int, int]) -> list[int]:
        my_orders_eaten = []
        i = 0
        while i < len(self.asks) and self.asks[i].total_amount() <= market_order[0]:
            market_order[0] -= self.asks[i].total_amount() # self.asks[i].execute_market_order(market_order[0])
            if self.asks[i].my_order_id != None: # if me executed
                my_orders_eaten.append(self.asks[i].my_order_id)
            i += 1

        if i == len(self.asks):
            self.asks = []
        else:
            _, my_id = self.asks[i].execute_market_order(market_order[0])
            if my_id != None:
                my_orders_eaten.append(my_id)
            del self.asks[:i]
        
        return my_orders_eaten
    
    def set_sell_market_order(self, market_order: Tuple[int, int]) -> list[int]:
        my_orders_eaten = []
        i = 0
        while i < len(self.bids) and self.bids[i].total_amount() <= market_order[0]:
            market_order[0] -= self.bids[i].total_amount() # self.bids[i].execute_market_order(market_order[0])
            if self.bids[i].my_order_id != None: # if me executed
                my_orders_eaten.append(self.bids[i].my_order_id)
            i += 1

        if i == len(self.bids):
            self.bids = []
        else:
            _, my_id = self.bids[i].execute_market_order(market_order[0])
            if my_id != None:
                my_orders_eaten.append(my_id)
            del self.bids[:i]
        
        return my_orders_eaten

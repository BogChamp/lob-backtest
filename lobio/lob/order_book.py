from typing import Sequence, Tuple, Self
from collections import defaultdict
from bisect import bisect_left

from .limit_order import LimitOrder, PRICE_TICK, AMOUNT_TICK
from .price_level import PriceLevel, Side, TraderId

PRETTY_INDENT_OB: int = 36

class OrderBook:
    """Order Book Simulator."""

    def __init__(
        self, bids: Sequence[PriceLevel] = None, asks: Sequence[PriceLevel] = None
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

    def mid_price(self):
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

    @staticmethod
    def matching_engine(
        price_levels: Sequence[PriceLevel], 
        limit_order: LimitOrder
    ) -> Tuple[Tuple[int, int], float, int]:
        """Match of newly arrived order with price levels.

        Args:
        ----
            price_levels (Sequence[PriceLevel]): price levels which price suit limit order's price.
            limit_order (LimitOrder): limit order which should be executed.

        Returns:
        -------
            Tuple[dict[TraderId, list[int]], float, int]:
            1.dict with keys of traders ids who exchanged quote and values equal to [quote, amount of base spent or recieved].
            2.remained quote for trade after partial execution. If equal 0, then order executed entirely.
            3.index of price levels, from which new price levels should be started after order execution.
        """
        matches_info = {TraderId.MM: [0, 0], TraderId.MARKET: [0, 0]}
        remain_amount = limit_order.quote
        #sign = 2 * limit_order.side - 1
        
        if not len(price_levels):
            return matches_info, remain_amount, 0

        for i, price_level in enumerate(price_levels):
            if (limit_order.side * price_level.base < limit_order.side * limit_order.base):
                break
            this_price = price_level.base
            remain_amount, cur_match_info = price_level.execute_limit_order(
                remain_amount
            )
            for k, v in cur_match_info.items():
                trader_info = matches_info[k]
                trader_info[0] += v
                trader_info[1] += -v * this_price
                matches_info[k] = trader_info

            if remain_amount == 0:
                break

        if price_levels[i].quote == 0:
            i += 1
        
        return matches_info[TraderId.MM], remain_amount, i

    def set_limit_order(self, limit_order: LimitOrder) -> Tuple[int, int]:
        """Adding of new limit order to a lob.

        Args:
        ----
            limit_order (LimitOrder): limit order needed to add.

        Raises:
        ------
            Exception: if wrong side specified. (not 0 buy or 1 sell)

        Returns:
        -------
            dict[TraderId, list[int]]: information about users, who traded quote, amount of quote and base traded per id.
        """
        #sign = 2 * limit_order.side - 1
        if limit_order.side == Side.BUY:
            price_levels = self.bids
            opposite_price_levels = self.asks
        elif limit_order.side == Side.SELL:
            price_levels = self.asks
            opposite_price_levels = self.bids
        else:
            raise Exception("WRONG SIDE")

        old_amount = limit_order.quote
        matches_info, remain_amount, p_l_eaten = self.matching_engine(
            opposite_price_levels, limit_order
        )

        if limit_order.side == Side.BUY:
            self.asks = self.asks[p_l_eaten:]
        else:
            self.bids = self.bids[p_l_eaten:]

        if remain_amount > 0:
            if remain_amount != old_amount:
                new_limit_order = LimitOrder(
                    limit_order.base,
                    remain_amount,
                    limit_order.side,
                    limit_order.trader_id,
                )
                price_levels.insert(0, PriceLevel(new_limit_order))
            else:
                index = bisect_left(
                    price_levels, limit_order.side * limit_order.base, key=lambda x: limit_order.side * x.base
                )
                if index == len(price_levels):
                    price_levels.append(PriceLevel(limit_order))
                elif price_levels[index].base == limit_order.base:
                    price_levels[index].add_limit_order(limit_order)
                else:
                    price_levels.insert(index, PriceLevel(limit_order))

        return matches_info

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
                LimitOrder(bid_raw[0], bid_raw[1], Side.BUY, TraderId.MARKET)
            )
            bids.append(bid)

        for ask_raw in asks_raw:
            ask = PriceLevel(
                LimitOrder(ask_raw[0], ask_raw[1], Side.SELL, TraderId.MARKET)
            )
            asks.append(ask)

        return cls(bids, asks)

    def update_price_levels(self, updates: Sequence[Tuple[int, int]], side: int): # Создаю лимит внутри, мб надо снаружи?
        """Update price level according to given side and prices with values of change.

        Args:
        ----
            updates (Sequence[Sequence[float]]): list of base-quote pairs.
            side (int): which side (asks or bids) needed to be change in LOB.

        Raises:
        ------
            Exception: if wrong side specified. (not 0 buy or 1 sell)
        """
        #sign = 2 * side - 1
        if side == Side.BUY:
            price_levels = self.bids
        elif side == Side.SELL:
            price_levels = self.asks
        else:
            raise Exception("Wrong side!")

        for update in updates:  # THINK ABOUT ORDER OF PRICE CHANGES
            price = update[0] #round(update[0], PRICE_TICK)
            amount = update[1] #round(update[1], AMOUNT_TICK)
            index = bisect_left(
                price_levels, side * price, key=lambda x: side * x.base
            )
            if index == len(price_levels):
                if amount > 0:
                    price_levels.append(
                        PriceLevel(LimitOrder(price, amount, side, TraderId.MARKET))
                    )
            elif price_levels[index].base == price:
                amount_change = amount - price_levels[index].quote
                price_levels[index].change_liquidity(amount_change, TraderId.MARKET)
                if price_levels[index].quote == 0:
                    del price_levels[index]
            else:
                if amount > 0:
                    price_levels.insert(
                        index,
                        PriceLevel(LimitOrder(price, amount, side, TraderId.MARKET)),
                    )

    def apply_historical_update(self, updates: Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]):
        """Perform changes on order book according to historical dynamic movements of LOB.

        Args:
        ----
            updates (Tuple[float, Sequence, Sequence]): timestamp, list of updates for bids as pairs base-quote,
            list of updates for asks as pair base-quote
        """
        bids_update = updates[1]
        asks_update = updates[2]

        self.update_price_levels(bids_update, Side.BUY)
        self.update_price_levels(asks_update, Side.SELL)

    def remove_bid_ask_intersection(self) -> Tuple[int, int]:
        #matches_info = {TraderId.MM: [0, 0], TraderId.MARKET: [0, 0]}
        # if self.bids[0].base >= self.asks[0].base:
        #     print(self.bids[:5], self.asks[:5])
        my_data = [0, 0]
        while self.bids[0].base >= self.asks[0].base:
            #print(self.bids[:5], self.asks[:5])
            limit_orders = self.bids[0].traders_order
            self.bids = self.bids[1:]
            for limit_order in limit_orders:
                cur_match_info = self.set_limit_order(limit_order)
                my_data[0] += cur_match_info[0]
                my_data[1] += cur_match_info[0]
            # print(self.bids[:5], self.asks[:5])
            # raise Exception
        return my_data
        

class OrderBookPrep(OrderBook):
    """Order Book class for data preparation for backtest.

    Args:
    ----
        OrderBook : Base Class of order book
    """

    def track_diff_side(
        self, update: Sequence[Sequence[float]], side: int
    ) -> Sequence[Sequence[float]]:
        """Creation of new incremental diffs, removing influence from trades.

        Args:
        ----
            update (Sequence[Sequence[float]]): list of base-quote pairs.
            side (int): which side (asks or bids) needed to be tracked.

        Raises:
        ------
            Exception: if wrong side specified. (not 0 buy or 1 sell)

        Returns:
        -------
            Sequence[Sequence[float]]: list of updated diffs.
        """
        diff_new = []
        # sign = 2 * side - 1
        if side == Side.BUY:
            price_levels = self.bids
        elif side == Side.SELL:
            price_levels = self.asks
        else:
            raise Exception("WRONG SIDE")

        for price_level in update:
            price = price_level[0]#round(price_level[0], PRICE_TICK)
            amount = price_level[1]#round(price_level[1], AMOUNT_TICK)
            index = bisect_left(price_levels, side * price, key=lambda x: side * x.base)
            if index == len(price_levels):
                if amount > 0:
                    price_levels.append(
                        PriceLevel(LimitOrder(price, amount, side, TraderId.MARKET))
                    )
                    diff_new.append([price, amount])
            elif price_levels[index].base == price:
                amount_change = amount - price_levels[index].quote #round(amount - price_levels[index].quote, AMOUNT_TICK)
                if amount_change != 0:
                    diff_new.append([price, amount_change])
                    price_levels[index].change_liquidity(amount_change, TraderId.MARKET)
                    if price_levels[index].quote == 0:
                        del price_levels[index]
            else:
                if amount > 0:
                    price_levels.insert(
                        index,
                        PriceLevel(LimitOrder(price, amount, side, TraderId.MARKET)),
                    )
                    diff_new.append([price, amount])
        return diff_new

    def track_diff(
        self, diff: Tuple[float, Sequence, Sequence]
    ) -> Tuple[float, Sequence[Sequence[float]], Sequence[Sequence[float]]]:
        """Tracking difference of state between historical snapshot and local one after trades applied.

        Args:
        ----
            diff (Tuple[float, Sequence, Sequence]): timestamp in nanoseconds, list of updates for bids as pairs base-quote,
            list of updates for asks as pair base-quote
        Returns:
            Tuple[float, Sequence, Sequence]: timestamp in nanoseconds, new bids update without trades influence,
            new asks update without trades influence
        """
        bids_update = diff[1]
        asks_update = diff[2]

        bids_diff_new = self.track_diff_side(bids_update, Side.BUY)
        asks_diff_new = self.track_diff_side(asks_update, Side.SELL)

        return (diff[0], bids_diff_new, asks_diff_new)

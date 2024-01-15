from typing import Sequence, Tuple, Self
from collections import defaultdict
from bisect import bisect_left

from .limit_order import LimitOrder, PRICE_TICK, AMOUNT_TICK
from .price_level import PriceLevel
from enum import IntEnum

PRETTY_INDENT_OB: int = 36


class Side(IntEnum):
    """Class for LOB side: 0 if BUY, 1 if sell."""

    BUY = 0
    SELL = 1


class TraderId(IntEnum):
    """Traders id: market, us (MM), or particular trader."""

    MARKET: int = 0
    MM: int = 1


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

    @staticmethod
    def matching_engine(
        price_levels: Sequence[PriceLevel], limit_order: LimitOrder
    ) -> Tuple[defaultdict[list], float, int]:
        """Match of newly arrived order with price levels.

        Args:
        ----
            price_levels (Sequence[PriceLevel]): price levels which price suit limit order's price.
            limit_order (LimitOrder): limit order which should be executed.

        Returns:
        -------
            Tuple[defaultdict[list], float, int]:
            1.dict with keys of traders ids who exchanged quote and values equal to [quote, amount of base spent or recieved].
            2.remained quote for trade after partial execution. If equal 0, then order executed entirely.
            3.index of price levels, from which new price levels should be started after order execution.
        """
        matches_info = defaultdict(list)
        remain_amount = limit_order.quote
        sign = 2 * limit_order.side - 1
        
        for i, price_level in enumerate(price_levels):
            if (sign * price_level.base < sign * limit_order.base) or (
                remain_amount == 0
            ):
                break
            this_price = price_level.base
            remain_amount, cur_match_info = price_level.execute_limit_order(
                remain_amount
            )
            for k, v in cur_match_info.items():
                trader_info = matches_info[k]
                if len(trader_info):
                    trader_info[0] += v
                    trader_info[1] += v * this_price
                else:
                    trader_info = [v, v * this_price]
                matches_info[k] = trader_info

        if price_levels[i].quote == 0:
            i += 1
        
        return matches_info, remain_amount, i

    def set_limit_order(self, limit_order: LimitOrder) -> defaultdict[list]:
        """Adding of new limit order to a lob.

        Args:
        ----
            limit_order (LimitOrder): limit order needed to add.

        Raises:
        ------
            Exception: if wrong side specified. (not 0 buy or 1 sell)

        Returns:
        -------
            defaultdict[list]: information about users, who traded quote, amount of quote and base traded per id.
        """
        sign = 2 * limit_order.side - 1
        if sign == -1:
            price_levels = self.bids
            opposite_price_levels = self.asks
        elif sign == 1:
            price_levels = self.asks
            opposite_price_levels = self.bids
        else:
            raise Exception("WRONG SIDE")

        old_amount = limit_order.quote
        matches_info, remain_amount, p_l_eaten = self.matching_engine(
            opposite_price_levels, limit_order
        )

        if sign == -1:
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
                    price_levels, sign * limit_order.base, key=lambda x: sign * x.base
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

    def update_limit_orders(self, updates: Sequence[Sequence[float]], side: int):
        """Update price level according to given side and prices with values of change.

        Args:
        ----
            updates (Sequence[Sequence[float]]): list of base-quote pairs.
            side (int): which side (asks or bids) needed to be change in LOB.

        Raises:
        ------
            Exception: if wrong side specified. (not 0 buy or 1 sell)
        """
        sign = 2 * side - 1
        if side == Side.BUY:
            price_levels = self.bids
        elif side == Side.SELL:
            price_levels = self.asks
        else:
            raise Exception("Wrong side!")

        for update in updates:  # THINK ABOUT ORDER OF PRICE CHANGES
            index = bisect_left(
                price_levels, sign * update[0], key=lambda x: sign * x.base
            )
            if index == len(price_levels):
                if update[1] > 0:
                    self.set_limit_order(
                        LimitOrder(update[0], update[1], side, TraderId.MARKET)
                    )
            elif price_levels[index].price == update[0]:
                price_levels[index].change_liquidity(update[1], TraderId.MARKET)
                if price_levels[index].quote == 0:
                    del price_levels[index]
            else:
                if update[1] > 0:
                    self.set_limit_order(
                        LimitOrder(update[0], update[1], Side.BUY, TraderId.MARKET)
                    )

    def apply_historical_update(self, updates: Tuple[float, Sequence, Sequence]):
        """Perform changes on order book according to historical dynamic movements of LOB.

        Args:
        ----
            updates (Tuple[float, Sequence, Sequence]): timestamp, list of updates for bids as pairs base-quote,
            list of updates for asks as pair base-quote
        """
        bids_update = updates[1]
        asks_update = updates[2]

        self.update_limit_orders(bids_update, Side.BUY)
        self.update_limit_orders(asks_update, Side.SELL)


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
        sign = 2 * side - 1
        if sign == -1:
            price_levels = self.bids
        elif sign == 1:
            price_levels = self.asks
        else:
            raise Exception("WRONG SIDE")

        for price_level in update:
            price = round(price_level[0], PRICE_TICK)
            amount = round(price_level[1], AMOUNT_TICK)
            index = bisect_left(price_levels, sign * price, key=lambda x: sign * x.base)
            if index == len(price_levels):
                if amount > 0:
                    price_levels.append(
                        PriceLevel(LimitOrder(price, amount, side, TraderId.MARKET))
                    )
                    diff_new.append([price, amount])
            elif price_levels[index].base == price:
                amount_change = round(amount - price_levels[index].quote, AMOUNT_TICK)
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

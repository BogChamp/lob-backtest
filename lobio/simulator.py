from typing import Sequence, Tuple
from .accounting.pnl_counter import PnL_Counter
from .lob.order_book import OrderBook, TraderId, Order, PRICE_TICK, AMOUNT_TICK, Side, OrderType
from .utils.utils import get_initial_order_book
from .model.avellaneda_stoikov_model import AvellanedaStoikov
from bisect import bisect_left
from tqdm import tqdm
from copy import deepcopy

class Simulator:
    """Class with implementatation of simulation of historical lob dynamics."""

    def __init__(
        self,
        diffs: Sequence[Tuple[int, Sequence[int], Sequence[int]]],
        orders: Sequence[Sequence[Tuple[int, Order]]],
        init_lob,
        model: AvellanedaStoikov,
        pnl_counter: PnL_Counter,
        time_end: int,
    ):
        """Set parameters and data for backtest.

        Args:
        ----
            diffs (Sequence[Tuple[float, Sequence[float], Sequence[float]]]): diffs to be applied on LOB.
            trades (Sequence[Sequence[Tuple[float, LimitOrder]]]): historical trades distributed between diffs.
            Every bunch of trades applied before according diff.
            init_lob (dict): initial LOB state.
            model (AvellanedaStoikov): model for trading and setting orders.
            pnl_counter (PnL_Counter): class for counting PnL.
            time_end (float): final time in nanoseconds for trading simulation, after which simulation stops.
        """
        assert len(orders) == len(diffs)

        self.diffs = diffs
        self.orders = orders
        self.init_lob = init_lob
        self.model = model
        self.pnl_counter = pnl_counter
        self.time_end = time_end

    def _apply_historical_orders(self, orders: Sequence[Tuple[int, Order]], 
                                  order_book: OrderBook):

        for _, order in orders:
            if order.type == OrderType.LIMIT:
                order_book.set_limit_order(order)
            else:
                match_info = order_book.set_market_order(order)
                self.q += match_info[0]
                self.wealth += match_info[1]

    def run_maker(self, market_latency: int = 0, local_latency: int = 0) -> Tuple[int, int]:
        self.pnl_counter.reset()
        self.model.reset()

        self.order_book = get_initial_order_book(self.init_lob)
        orders = deepcopy(self.orders)

        self.q = 0
        self.wealth = 0

        for i, diff in enumerate(tqdm(self.diffs)):
            if diff[0] > self.time_end:
                break

            cur_orders = orders[i]

            #print("Create bids")
            my_bids, my_asks = self.model.bid_ask_limit_orders(
                self.order_book, diff[0] + market_latency, self.q
            )
            my_order_setting_time = diff[0] + market_latency + local_latency
            #print("log search")
            my_order_index = bisect_left(
                cur_orders, my_order_setting_time, key=lambda x: x[0]
            )
            orders_before = cur_orders[:my_order_index]
            orders_after = cur_orders[my_order_index:]
            #print("apply before")
            self._apply_historical_orders(orders_before, self.order_book)

            #last_trade_price = order_book.ask_price()
            #print("apply my")
            for my_order in my_bids + my_asks:
                self.order_book.set_limit_order(my_order)
            #print("apply after")
            self._apply_historical_orders(orders_after, self.order_book)
            #last_trade_price = order_book.ask_price()

            #print("apply diff")
            self.order_book.apply_historical_update(diff)
            #print("intersect")
            q_change, wealth_change = self.order_book.remove_bid_ask_intersection()
            self.q += q_change
            self.wealth += wealth_change
            self.pnl_counter.collect_statistic(self.q, self.order_book.mid_price())

        self.pnl_counter.collect_statistic(self.q, self.order_book.mid_price())

        return self.q, self.wealth
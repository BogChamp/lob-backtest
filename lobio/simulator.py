from typing import Sequence, Tuple
from .accounting.pnl_counter import PnL_Counter
from .lob.order_book import OrderBook, TraderId, LimitOrder, PRICE_TICK, AMOUNT_TICK
from .model.avellaneda_stoikov_model import AvellanedaStoikov
from bisect import bisect_left
from tqdm import tqdm
from copy import deepcopy

class Simulator:
    """Class with implementatation of simulation of historical lob dynamics."""

    def __init__(
        self,
        diffs: Sequence[Tuple[float, Sequence[float], Sequence[float]]],
        trades: Sequence[Sequence[Tuple[float, LimitOrder]]],
        init_lob: dict,
        model: AvellanedaStoikov,
        pnl_counter: PnL_Counter,
        time_end: float,
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
        assert len(trades) == len(diffs)

        self.diffs = diffs
        self.trades = trades
        self.init_lob = init_lob
        self.model = model
        self.pnl_counter = pnl_counter
        self.time_end = time_end

    def run(
        self, market_latency: int, local_latency: int
    ) -> Tuple[list[float], float, float]:
        """Run backtest with given latencies.

           Sequentially applies trades and diffs according to historical timestamps.
           Calcultes and returns statistics.

        Args:
        ----
            market_latency (int): latency on server, how long exchange takes time to process data.
            local_latency (int): local latency, how much time needed for interaction with server.

        Returns:
        -------
            Tuple[list[float], float, float]:
            1. PnL graph
            2. Our final quote amount
            3. Our final base amount.
        """
        self.pnl_counter.reset()
        self.model.reset()

        order_book = OrderBook.create_lob_init(self.init_lob)
        trades = deepcopy(self.trades)

        last_trade_price = order_book.ask_price()
        pnl_history = [0.0]
        q = 0.0
        wealth = 0.0

        for i, diff in enumerate(tqdm(self.diffs)):
            if diff[0] > self.time_end:
                break

            cur_trades = trades[i]

            my_bids, my_asks = self.model.bid_ask_limit_orders(
                order_book, diff[0] + market_latency, q
            )
            my_order_setting_time = diff[0] + market_latency + local_latency
            my_order_index = bisect_left(
                cur_trades, my_order_setting_time, key=lambda x: x[0]
            )

            trades_before = cur_trades[:my_order_index]
            trades_after = cur_trades[my_order_index:]

            q_change, wealth_change = self.__apply_historical_trades(trades_before, order_book, last_trade_price)
            q += q_change
            wealth += wealth_change
            last_trade_price = order_book.ask_price()

            for my_order in my_bids + my_asks:
                match_info = order_book.set_limit_order(my_order)
                sign = 2 * my_order.side - 1
                for _, v in match_info.items():
                    q += -sign * v[0]
                    wealth += sign * v[1]
                self.pnl_counter.change_pnl(last_trade_price, order_book.ask_price(), q)
                last_trade_price = order_book.ask_price()

            q_change, wealth_change = self.__apply_historical_trades(trades_after, order_book, last_trade_price)
            q += q_change
            wealth += wealth_change
            last_trade_price = order_book.ask_price()

            q = round(q, AMOUNT_TICK)
            wealth = round(wealth, PRICE_TICK)
            pnl_history.append(self.pnl_counter.pnl)
            order_book.apply_historical_update(diff)

        self.pnl_counter.change_pnl(last_trade_price, order_book.ask_price(), q)
        pnl_history.append(self.pnl_counter.pnl)
        self.order_book = order_book

        return pnl_history, q, wealth

    def __apply_historical_trades(self, trades: Sequence[Tuple[float, LimitOrder]], 
                                  order_book: OrderBook,
                                  last_trade_price: float):

        q = 0
        wealth = 0
        for _, limit_order in trades:
            match_info = order_book.set_limit_order(limit_order)
            sign = 2 * limit_order.side - 1
            my_match_info = match_info[TraderId.MM]
            if len(my_match_info):
                q += sign * my_match_info[0]
                wealth += -sign * my_match_info[1]
                self.pnl_counter.change_pnl(
                    last_trade_price, order_book.ask_price(), q
                )
                last_trade_price = order_book.ask_price()
        
        return q, wealth

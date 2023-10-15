from typing import Sequence
from PnLCounter import PnL_Counter
from OrderBook import OrderBook, MM_ID, PRICE_TICK, AMOUNT_TICK
from ASmodel import AvellanedaStoikov
from bisect import bisect_left
from tqdm import tqdm

class Simulator:
    def __init__(self, diffs: Sequence[dict], trades: Sequence[dict], init_lob: dict,
                 model: AvellanedaStoikov, pnl_counter: PnL_Counter, time_end: float):
        self.diffs = diffs
        self.trades = trades
        self.init_lob = init_lob
        self.model = model
        self.pnl_counter = pnl_counter
        self.time_end = time_end

    def run(self, market_latency: int, local_latency: int):
        order_book = OrderBook.create_lob_init(self.init_lob)

        last_trade_price = order_book.ask_price()
        pnl_history = [0]
        q = 0
        wealth = 0

        for i, diff in enumerate(tqdm(self.diffs)):
            if diff[0] > self.time_end:
                break
            
            cur_trades = self.trades[i]
            my_bid, my_ask = self.model.bid_ask_limit_orders(order_book, diff[0]+market_latency, q)
            market_interaction = diff[0]+market_latency+local_latency
            my_order_index = bisect_left(cur_trades, market_interaction, key=lambda x: x[0])
            trades_before = cur_trades[:my_order_index]
            trades_after = cur_trades[my_order_index:]

            for ts, limit_order in trades_before:
                match_info = order_book.set_limit_order(limit_order)
                sign = 2 * limit_order.side - 1
                my_match_info = match_info[MM_ID]
                if len(my_match_info):
                    q += sign * my_match_info[0]
                    wealth += -sign * my_match_info[1]
                    self.pnl_counter.change_pnl(last_trade_price, order_book.ask_price(), q)
                    last_trade_price = order_book.ask_price()

            for my_order in [my_bid, my_ask]:
                match_info = order_book.set_limit_order(my_order)
                sign = 2 * my_order.side - 1
                for k, v in match_info.items():
                    q += -sign * v[0]
                    wealth += sign * v[1]
                self.pnl_counter.change_pnl(last_trade_price, order_book.ask_price(), q)
                last_trade_price = order_book.ask_price()
            
            for ts, limit_order in trades_after:
                match_info = order_book.set_limit_order(limit_order)
                sign = 2 * limit_order.side - 1
                my_match_info = match_info[MM_ID]
                if len(my_match_info):
                    q += sign * my_match_info[0]
                    wealth += -sign * my_match_info[1]
                    last_trade_price = order_book.ask_price()
                    self.pnl_counter.change_pnl(last_trade_price, order_book.ask_price(), q)
                    last_trade_price = order_book.ask_price()

            q = round(q, AMOUNT_TICK)
            wealth = round(wealth, PRICE_TICK)
            pnl_history.append(self.pnl_counter.pnl)
            order_book.apply_historical_update(diff)
        
        self.pnl_counter.change_pnl(last_trade_price, order_book.ask_price(), q)
        pnl_history.append(self.pnl_counter.pnl)

        return pnl_history, q, wealth

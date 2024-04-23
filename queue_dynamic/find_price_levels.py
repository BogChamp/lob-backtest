from tqdm import tqdm
from bisect import bisect_left
from copy import deepcopy
import numpy as np
from loguru import logger

from lobio.utils.utils import get_initial_order_book_trunc, group_diffs, group_orders
from lobio.lob.limit_order import Side, OrderType

if __name__ == "__main__":
    init_lob_prepared_file = "./data/init_lob_prepared.npy"
    diffs_prepared_file = "./data/diffs_prepared.npy"
    orders_prepared_file = "./data/orders_prepared.npy"

    with open(init_lob_prepared_file, 'rb') as file:
        init_lob = np.load(file)
    with open(diffs_prepared_file, 'rb') as file:
        diffs = np.load(file)
    with open(orders_prepared_file, 'rb') as file:
        orders = np.load(file)

    logger.info('Group data!')
    diffs_grouped = group_diffs(diffs)
    orders_per_diff = group_orders(orders, len(diffs_grouped))

    logger.info('Find appropriate price levels!')
    ob = get_initial_order_book_trunc(init_lob)
    MAX_PAST_DIFFS = 30
    last_bid_states = []
    last_ask_states = []
    last_pl_removed = []
    possible_ts = []
    for i, diff in enumerate(tqdm(diffs_grouped)):
        last_bid_states.append((i, deepcopy(ob.bids)))
        last_ask_states.append((i, deepcopy(ob.asks)))
        if len(last_bid_states) > MAX_PAST_DIFFS:
            del last_bid_states[0]
        if len(last_ask_states) > MAX_PAST_DIFFS:
            del last_ask_states[0]

        cur_pl_removed = set()
        cur_orders = orders_per_diff[i]
        for _, order in cur_orders:
            if order.type == OrderType.MARKET:
                if len(ob.bids): # To track removed not best price leves
                    prev_best_bid = ob.bids[0][0]
                if len(ob.asks): # To track removed not best price leves
                    prev_best_ask = ob.asks[0][0]
                ob.set_market_order([order.quote, order.side])
                if order.side == Side.BUY and len(ob.asks):
                    if ob.asks[0][0] == order.base: # if order not ate whole price level
                        is_found = False
                        ind = bisect_left(last_ask_states[-1][1], order.base, key=lambda x: x[0])
                        if ind != len(last_ask_states[-1][1]) and last_ask_states[-1][1][ind][0] == order.base:
                            for j, (ts, last_ask) in enumerate(last_ask_states[-2::-1]):
                                if order.base in last_pl_removed[-j-1]:
                                    is_found = True
                                    if j != 0: # to delete all price levels which live only 1 diff
                                        possible_ts.append((ts+1, order.base, Side.SELL))
                                    break
                                ind = bisect_left(last_ask, order.base, key=lambda x: x[0])
                                if ind == len(last_ask) or last_ask[ind][0] != order.base:
                                    is_found = True
                                    if j != 0: # to delete all price levels which live only 1 diff
                                        possible_ts.append((ts+1, order.base, Side.SELL))
                                    break
                            if not is_found:
                                possible_ts.append((ts, order.base, Side.SELL))
                    else:
                        cur_best_ask = ob.asks[0][0]
                        while cur_best_ask > prev_best_ask:
                            cur_pl_removed.add(prev_best_ask)
                            prev_best_ask += 1

                elif len(ob.bids):
                    if ob.bids[0][0] == order.base: # if order not ate whole price level
                        is_found = False
                        ind = bisect_left(last_bid_states[-1][1], -order.base, key=lambda x: -x[0])
                        if ind != len(last_bid_states[-1][1]) and last_bid_states[-1][1][ind][0] == order.base:
                            for j, (ts, last_bid) in enumerate(last_bid_states[-2::-1]):
                                if order.base in last_pl_removed[-j-1]:
                                    is_found = True
                                    if j != 0:
                                        possible_ts.append((ts+1, order.base, Side.BUY))
                                    break
                                ind = bisect_left(last_bid, -order.base, key=lambda x: -x[0])
                                if ind == len(last_bid) or last_bid[ind][0] != order.base:
                                    is_found = True
                                    if j != 0:
                                        possible_ts.append((ts+1, order.base, Side.BUY))
                                    break
                            if not is_found:
                                possible_ts.append((ts, order.base, Side.BUY))
                    else:
                        cur_best_bid = ob.bids[0][0]
                        while cur_best_bid < prev_best_bid:
                            cur_pl_removed.add(prev_best_bid)
                            prev_best_bid -= 1
            else:
                ob.set_limit_order([order.base, order.quote, order.side])
    
        last_pl_removed.append(cur_pl_removed)
        if len(last_pl_removed) > MAX_PAST_DIFFS:
            del last_pl_removed[0]
        ob.apply_historical_update(diff)
    
    possible_ts.sort(key=lambda x: (x[0], x[1], x[2]))

    squeezed_ts = []
    prev_info = None
    for info in possible_ts:
        if info != prev_info:
            squeezed_ts.append(info)
            prev_info = info
    
    with open('./data/price_level_to_enter.npy', 'wb') as f:
        np.save(f, squeezed_ts)

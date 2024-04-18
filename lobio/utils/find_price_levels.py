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
    possible_ts = []
    for i, diff in enumerate(tqdm(diffs_grouped)):
        last_bid_states.append((i, deepcopy(ob.bids)))
        last_ask_states.append((i, deepcopy(ob.asks)))
        if len(last_bid_states) > MAX_PAST_DIFFS:
            del last_bid_states[0]
        if len(last_ask_states) > MAX_PAST_DIFFS:
            del last_ask_states[0]

        cur_orders = orders_per_diff[i]
        for _, order in cur_orders:
            if order.type == OrderType.MARKET:
                ob.set_market_order([order.quote, order.side])
                if order.side == Side.BUY and len(ob.asks):
                    if ob.asks[0][0] == order.base: # if order not ate whole price level
                        is_found = False
                        ind = bisect_left(last_ask_states[-1][1], order.base, key=lambda x: x[0])
                        if ind == len(last_ask_states[-1][1]) or last_ask_states[-1][1][ind][0] != order.base:
                            is_found = True # to prevent adding - it is between diffs price level
                        else:
                            for ts, last_ask in last_ask_states[-2::-1]:
                                ind = bisect_left(last_ask, order.base, key=lambda x: x[0])
                                if ind == len(last_ask) or last_ask[ind][0] != order.base:
                                    possible_ts.append((ts+1, order.base, Side.SELL))
                                    is_found = True
                                    break
                        if not is_found:
                            possible_ts.append((ts, order.base, Side.SELL))
                elif len(ob.bids):
                    if ob.bids[0][0] == order.base: # if order not ate whole price level
                        is_found = False
                        ind = bisect_left(last_bid_states[-1][1], -order.base, key=lambda x: -x[0])
                        if ind == len(last_bid_states[-1][1]) or last_bid_states[-1][1][ind][0] != order.base:
                            is_found = True
                        else:
                            for ts, last_bid in last_bid_states[-2::-1]:
                                ind = bisect_left(last_bid, -order.base, key=lambda x: -x[0])
                                if ind == len(last_bid) or last_bid[ind][0] != order.base:
                                    possible_ts.append((ts+1, order.base, Side.BUY))
                                    is_found = True
                                    break
                        if not is_found:
                            possible_ts.append((ts, order.base, Side.BUY))
            else:
                ob.set_limit_order([order.base, order.quote, order.side])
        
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
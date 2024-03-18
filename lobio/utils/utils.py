import numpy as np
import polars as pl
from typing import Sequence, Tuple
from copy import deepcopy
from tqdm import tqdm

from lobio.lob.limit_order import LimitOrder, AMOUNT_TICK
from lobio.lob.price_level import Side, TraderId
from lobio.lob.order_book import OrderBook, OrderBookPrep

def calculate_ticker_changes(tickers_raw: dict) -> Sequence[Tuple[float, float, int]]:
    tickers_chg = []
    for i, ticker in enumerate(tickers_raw[1:]):
        if ticker['b'] != tickers_raw[i]['b']:
            tickers_chg.append([tickers_raw[i]['b'], tickers_raw[i]['B'], Side.BUY])
        elif ticker['B'] != tickers_raw[i]['B']:
            tickers_chg.append([tickers_raw[i]['b'], round(tickers_raw[i]['B'] - ticker['B'], AMOUNT_TICK), Side.BUY])
        
        if ticker['a'] != tickers_raw[i]['a']:
            tickers_chg.append([tickers_raw[i]['a'], tickers_raw[i]['A'], Side.SELL])
        elif ticker['A'] != tickers_raw[i]['A']:
            tickers_chg.append([tickers_raw[i]['a'], round(tickers_raw[i]['A'] - ticker['A'], AMOUNT_TICK), Side.SELL])
    
    return tickers_chg

def find_ticker_aggtrades_matches(tickers_raw: Sequence[dict], 
                                  aggtrades_raw: Sequence[dict]) -> int:
    aggtrade_ind = 0
    prev_ts_bid = 0
    prev_ts_ask = 0

    prev_b = tickers_raw[0]['b']
    prev_B = tickers_raw[0]['B']
    prev_a = tickers_raw[0]['a']
    prev_A = tickers_raw[0]['A']
    tickers_per_trade = []
    current_tickers = []
    for ticker in tickers_raw:
        cur_b = ticker['b']
        cur_B = ticker['B']
        cur_a = ticker['a']
        cur_A = ticker['A']

        if aggtrades_raw[aggtrade_ind]['m']:
            if cur_b != prev_b or cur_B != prev_B:
                if cur_b != prev_b:
                    quote_change = prev_B
                    price_change = prev_b
                else:
                    quote_change = round(prev_B - cur_B, AMOUNT_TICK)
                    price_change = cur_b

                current_tickers.append([price_change, quote_change, Side.BUY])

                if aggtrades_raw[aggtrade_ind]['p'] == price_change and aggtrades_raw[aggtrade_ind]['q'] == quote_change:
                    prev_ts_bid = aggtrades_raw[aggtrade_ind]['T']
                    tickers_per_trade.append(current_tickers)
                    current_tickers = []
                    aggtrade_ind += 1
                    while aggtrade_ind < len(aggtrades_raw) and \
                            aggtrades_raw[aggtrade_ind]['m'] and \
                            aggtrades_raw[aggtrade_ind]['T'] == prev_ts_bid:
                        aggtrade_ind += 1
                    
                    if aggtrade_ind >= len(aggtrades_raw):
                        break
        else:
            if cur_a != prev_a or cur_A != prev_A:
                if cur_a != prev_a:
                    quote_change = prev_A
                    price_change = prev_a
                else:
                    quote_change = round(prev_A - cur_A, AMOUNT_TICK)
                    price_change = cur_a

                current_tickers.append([price_change, quote_change, Side.SELL])

                if aggtrades_raw[aggtrade_ind]['p'] == price_change and aggtrades_raw[aggtrade_ind]['q'] == quote_change:
                    prev_ts_ask = aggtrades_raw[aggtrade_ind]['T']
                    tickers_per_trade.append(current_tickers)
                    current_tickers = []
                    aggtrade_ind += 1
                    while aggtrade_ind < len(aggtrades_raw) and \
                            not aggtrades_raw[aggtrade_ind]['m'] and \
                            aggtrades_raw[aggtrade_ind]['T'] == prev_ts_ask:
                        aggtrade_ind += 1
                    
                    if aggtrade_ind >= len(aggtrades_raw):
                        break

        prev_b = cur_b
        prev_B = cur_B
        prev_a = cur_a
        prev_A = cur_A
    
    return aggtrade_ind

def get_grouped_diff(diff_info: Tuple[int, 
                                      Sequence[int], 
                                      Sequence[int], 
                                      Sequence[int]]) -> Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]:
        ts = diff_info[0]
        bases = diff_info[1]
        quotes = diff_info[2]
        sides = diff_info[3]
        bids = []
        asks = []

        for i, side in enumerate(sides):
            if side == Side.BUY:
                bids.append((bases[i], quotes[i]))
            else:
                asks.append((bases[i], quotes[i]))

        return (ts, bids, asks)

def group_diffs(diffs_prepared: pl.DataFrame) -> Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]]:
    diffs_grouped_T = diffs_prepared.group_by('T').agg(pl.col('base'), pl.col('quote'), pl.col('side')).sort(by='T')

    diffs_grouped = []
    for diff in diffs_grouped_T.iter_rows():
        diffs_grouped.append(get_grouped_diff(diff))
    
    return diffs_grouped

def get_initial_order_book_prep(init_lob_data: np.ndarray[int]) -> OrderBookPrep:
    ts, init_lob = init_lob_data[-1][0], init_lob_data[:-1]
    bids = init_lob[:, :2]
    asks = init_lob[:, 2:]
    init_lob_dict = {"lastUpdateId": ts, "bids": bids, "asks": asks}
    order_book = OrderBookPrep.create_lob_init(init_lob_dict)
    return order_book

def get_initial_order_book(init_lob_data: np.ndarray[int]) -> OrderBook:
    ts, init_lob = init_lob_data[-1][0], init_lob_data[:-1]
    bids = init_lob[:, :2]
    asks = init_lob[:, 2:]
    init_lob_dict = {"lastUpdateId": ts, "bids": bids, "asks": asks}
    order_book = OrderBook.create_lob_init(init_lob_dict)
    return order_book

def check_if_sorted(diffs_grouped: Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]]):
    for diff in diffs_grouped:
        bids = diff[1]
        asks = diff[2]
        if len(bids):
            prev_price = bids[0][0]
            for bid in bids[1:]:
                assert bid[0] < prev_price
                prev_price = bid[0]
        if len(asks):
            prev_price = asks[0][0]
            for ask in asks[1:]:
                assert ask[0] > prev_price
                prev_price = ask[0]

def find_price_in_asks(asks, price):
    price_level = []
    for ask in asks:
        if ask[0] == price:
            return (True, ask)
        elif ask[0] > price:
            return (False, price_level)
    return (False, price_level)

def find_price_in_bids(bids, price):
    price_level = []
    for bid in bids:
        if bid[0] == price:
            return (True, bid)
        elif bid[0] < price:
            return (False, price_level)
    return (False, price_level)

def group_historical_trades(aggtrades_raw, diffs_grouped) -> Sequence[Sequence[Tuple[int, int, int, int]]]:
    trades_per_diff = []
    trades_index = 0

    trades_after_prev_diff = []
    cur_trade = aggtrades_raw[trades_index]
    for i, diff in enumerate(diffs_grouped):
        time_to = diff[0]

        trades_before_diff = trades_after_prev_diff
        while cur_trade[0] < time_to and trades_index < len(aggtrades_raw): 
            trades_before_diff.append(cur_trade)
            trades_index += 1
            if trades_index >= len(aggtrades_raw):
                break
            cur_trade = aggtrades_raw[trades_index]

        trades_after_prev_diff = []
        while cur_trade[0] == time_to and trades_index < len(aggtrades_raw):
            if cur_trade[3] == Side.BUY:
                asks = diff[2] # should be sorted
                asks_next = diffs_grouped[i+1][2] # should be sorted
                cur_info = find_price_in_asks(asks, cur_trade[1])
                next_info = find_price_in_asks(asks_next,  cur_trade[1])
                if not cur_info[0]:
                    trades_after_prev_diff.append(cur_trade)
                elif not next_info[0]:
                    trades_before_diff.append(cur_trade)
                else:
                    if cur_info[1][1] == 0:
                        trades_before_diff.append(cur_trade)
                    else:
                        trades_after_prev_diff.append(cur_trade)
                    
                if not cur_info[0] and not next_info[0]:
                    print("Hard to detect place of trade", i, cur_trade)
            else:
                bids = diff[1] # should be sorted
                bids_next = diffs_grouped[i+1][1] # should be sorted
                cur_info = find_price_in_bids(bids, cur_trade[1])
                next_info = find_price_in_bids(bids_next,  cur_trade[1])
                if not cur_info[0]:
                    trades_after_prev_diff.append(cur_trade)
                elif not next_info[0]:
                    trades_before_diff.append(cur_trade)
                else:
                    if cur_info[1][1] == 0:
                        trades_before_diff.append(cur_trade)
                    else:
                        trades_after_prev_diff.append(cur_trade)
                    
                if not cur_info[0] and not next_info[0]:
                    print("Hard to detect place of trade", i, cur_trade)

            trades_index += 1
            if trades_index >= len(aggtrades_raw):
                break
            cur_trade = aggtrades_raw[trades_index]

        trades_per_diff.append(trades_before_diff)

    return trades_per_diff

def find_unseen_dynamic_of_lob(init_lob: np.ndarray[int], 
                               trades_per_diff: Sequence[Sequence[Tuple[int, int, int, int]]],
                               diffs_grouped: Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]]) -> \
                                Tuple[Sequence[Tuple[Tuple[int, int], int, int, int]], Sequence[Tuple[Tuple[int, int], int, int, int]]]:
    ob = get_initial_order_book(init_lob)
    additional_data = [] # (transaction time, order of event, diff number) p q side
    trades_prepared = []

    between_diffs_index = 0

    trades_copy = deepcopy(trades_per_diff)
    diffs_grouped_preprocess = [(init_lob[-1][0], [], [])] + diffs_grouped
    for i, diff in enumerate(tqdm(diffs_grouped_preprocess[1:])):
        cur_trades = trades_copy[i]
        
        cur_bids = [(diffs_grouped_preprocess[i][0], 0)] # ts, price
        cur_asks = [(diffs_grouped_preprocess[i][0], np.inf)]
        for trade in cur_trades:
            bids = ob.bids
            asks = ob.asks
            if trade[3] == Side.SELL:
                unseen_ts = cur_bids[0][0] # earliest time for unseen change in ob
                if trade[1] > bids[0].base:
                    j = 0   # check if trade on opposite side's price
                    updates = []
                    while trade[1] >= asks[j].base:
                        additional_data.append([(unseen_ts, between_diffs_index, i), asks[j].base, -asks[j].quote, Side.SELL])
                        updates.append([asks[j].base, 0])
                        between_diffs_index += 1
                        j += 1
                    ob.update_price_levels(updates, Side.SELL)

                    for ts_price in cur_bids[::-1]:
                        if trade[1] > ts_price[1]:
                            unseen_ts = ts_price[0]
                    additional_data.append([(unseen_ts, between_diffs_index, i), trade[1], trade[2], Side.BUY])
                    ob.set_limit_order(LimitOrder(trade[1], trade[2], Side.BUY, TraderId.MARKET))
                    between_diffs_index += 1
                elif trade[1] < bids[0].base:
                    j = 0
                    updates = []
                    while trade[1] < bids[j].base:
                        additional_data.append([(unseen_ts, between_diffs_index, i), bids[j].base, -bids[j].quote, Side.BUY])
                        updates.append([bids[j].base, 0])
                        between_diffs_index += 1
                        j += 1
                    ob.update_price_levels(updates, Side.BUY)

                    if trade[1] == bids[0].base:
                        if trade[2] > bids[0].quote:
                            additional_data.append([(unseen_ts, between_diffs_index, i), trade[1], trade[2]-bids[0].quote, Side.BUY])
                            ob.set_limit_order(LimitOrder(trade[1], trade[2]-bids[0].quote, Side.BUY, TraderId.MARKET))
                            between_diffs_index += 1
                    else:
                        additional_data.append([(unseen_ts, between_diffs_index, i), trade[1], trade[2], Side.BUY])
                        ob.set_limit_order(LimitOrder(trade[1], trade[2], Side.BUY, TraderId.MARKET))
                        between_diffs_index += 1
                elif trade[2] > bids[0].quote:
                    additional_data.append([(unseen_ts, between_diffs_index, i), trade[1], trade[2]-bids[0].quote, Side.BUY])
                    ob.set_limit_order(LimitOrder(trade[1], trade[2]-bids[0].quote, Side.BUY, TraderId.MARKET))
                    between_diffs_index += 1

                if trade[1] > cur_bids[-1][1]:
                    cur_bids.append((trade[0], trade[1]))
                elif trade[1] < cur_bids[-1][1]:
                    for ind, ts_price in enumerate(cur_bids[1:]):
                        if trade[1] < ts_price[1]:
                            cur_bids[ind+1] = (trade[0], trade[1])
                            cur_bids = cur_bids[:ind+2]
                            break
            else:
                unseen_ts = cur_asks[0][0]
                if trade[1] < asks[0].base:
                    j = 0   # check if trade on opposite side's price
                    updates = []
                    while trade[1] <= bids[j].base:
                        additional_data.append([(unseen_ts, between_diffs_index, i), bids[j].base, -bids[j].quote, Side.BUY])
                        updates.append([bids[j].base, 0])
                        between_diffs_index += 1
                        j += 1
                    ob.update_price_levels(updates, Side.BUY)

                    for ts_price in cur_asks[::-1]:
                        if trade[1] < ts_price[1]:
                            unseen_ts = ts_price[0]
                    additional_data.append([(unseen_ts, between_diffs_index, i), trade[1], trade[2], Side.SELL])
                    ob.set_limit_order(LimitOrder(trade[1], trade[2], Side.SELL, TraderId.MARKET))
                    between_diffs_index += 1
                elif trade[1] > asks[0].base:
                    j = 0
                    updates = []
                    while trade[1] > asks[j].base:
                        additional_data.append([(unseen_ts, between_diffs_index, i), asks[j].base, -asks[j].quote, Side.SELL])
                        updates.append([asks[j].base, -asks[j].quote])
                        between_diffs_index += 1
                        j += 1
                    ob.update_price_levels(updates, Side.SELL)

                    if trade[1] == asks[0].base:
                        if trade[2] > asks[0].quote:
                            additional_data.append([(unseen_ts, between_diffs_index, i), trade[1], trade[2]-asks[0].quote, Side.SELL])
                            ob.set_limit_order(LimitOrder(trade[1], trade[2]-asks[0].quote, Side.SELL, TraderId.MARKET))
                            between_diffs_index += 1
                    else:
                        additional_data.append([(unseen_ts, between_diffs_index, i), trade[1], trade[2], Side.SELL])
                        ob.set_limit_order(LimitOrder(trade[1], trade[2], Side.SELL, TraderId.MARKET))
                        between_diffs_index += 1
                elif trade[2] > asks[0].quote:
                    additional_data.append([(unseen_ts, between_diffs_index, i), trade[1], trade[2]-asks[0].quote, Side.SELL])
                    ob.set_limit_order(LimitOrder(trade[1], trade[2]-asks[0].quote, Side.SELL, TraderId.MARKET))
                    between_diffs_index += 1

                if trade[1] < cur_asks[-1][1]:
                    cur_asks.append((trade[0], trade[1]))
                elif trade[1] > cur_asks[-1][1]:
                    for ind, ts_price in enumerate(cur_asks[1:]):
                        if trade[1] > ts_price[1]:
                            cur_asks[ind+1] = (trade[0], trade[1])
                            cur_asks = cur_asks[:ind+2]
                            break

            trades_prepared.append([(trade[0], between_diffs_index, i), trade[1], trade[2], trade[3]])
            ob.set_limit_order(LimitOrder(trade[1], trade[2], trade[3], TraderId.MARKET))
            between_diffs_index += 1
            

        ob.apply_historical_update(diff)
    return trades_prepared, additional_data

def merge_orders(sorted_list1, sorted_list2) -> Sequence[Tuple[int, int, int, int, int]]:
    if len(sorted_list1) == 0:
        return sorted_list2
    elif len(sorted_list2) == 0:
        return sorted_list1

    ind1 = 0
    ind2 = 0

    merged_array = [(0, 0, 0, 0, 0)] * (len(sorted_list1) + len(sorted_list2)) # (ts, diff num, p, q, side)
    while ind1 < len(sorted_list1) and ind2 < len(sorted_list2):
        if sorted_list1[ind1][0][0] < sorted_list2[ind2][0][0]:
            merged_array[ind1+ind2] = (sorted_list1[ind1][0][0], sorted_list1[ind1][0][2], 
                                       sorted_list1[ind1][1], sorted_list1[ind1][2], sorted_list1[ind1][3])
            ind1 += 1
        elif sorted_list1[ind1][0][0] > sorted_list2[ind2][0][0]:
            merged_array[ind1+ind2] = (sorted_list2[ind2][0][0], sorted_list2[ind2][0][2], 
                                       sorted_list2[ind2][1], sorted_list2[ind2][2], sorted_list2[ind2][3])
            ind2 += 1
        elif sorted_list1[ind1][0][1] < sorted_list2[ind2][0][1]:
            merged_array[ind1+ind2] = (sorted_list1[ind1][0][0], sorted_list1[ind1][0][2],
                                       sorted_list1[ind1][1], sorted_list1[ind1][2], sorted_list1[ind1][3])
            ind1 += 1
        elif sorted_list1[ind1][0][1] > sorted_list2[ind2][0][1]:
            merged_array[ind1+ind2] = (sorted_list2[ind2][0][0], sorted_list2[ind2][0][2], 
                                       sorted_list2[ind2][1], sorted_list2[ind2][2], sorted_list2[ind2][3])
            ind2 += 1
        else:
            merged_array[ind1+ind2] = (sorted_list1[ind1][0][0], sorted_list1[ind1][0][2], 
                                       sorted_list1[ind1][1], sorted_list1[ind1][2], sorted_list1[ind1][3])
            ind1 += 1
            merged_array[ind1+ind2] = (sorted_list2[ind2][0][0], sorted_list2[ind2][0][2],
                                       sorted_list2[ind2][1], sorted_list2[ind2][2], sorted_list2[ind2][3])
            ind2 += 1
    
    while ind1 < len(sorted_list1):
        merged_array[ind1+ind2] = (sorted_list1[ind1][0][0], sorted_list1[ind1][0][2],
                                   sorted_list1[ind1][1], sorted_list1[ind1][2], sorted_list1[ind1][3])
        ind1 += 1
    
    while ind2 < len(sorted_list2):
        merged_array[ind1+ind2] = (sorted_list2[ind2][0][0], sorted_list2[ind2][0][2],
                                   sorted_list2[ind2][1], sorted_list2[ind2][2], sorted_list2[ind2][3])
        ind2 += 1
    
    return merged_array

def group_orders(orders_prepared_np: np.ndarray[int], max_diff_number: int) -> Sequence[Sequence[Tuple[int, int, int, int]]]:
    diffs_total = max_diff_number
    orders_per_diff = []
    order_ind = 0
    for i in range(diffs_total):
        cur_orders = []
        while order_ind < len(orders_prepared_np) and orders_prepared_np[order_ind][1] == i:
            cur_orders.append([orders_prepared_np[order_ind][0], orders_prepared_np[order_ind][2], 
                            orders_prepared_np[order_ind][3], orders_prepared_np[order_ind][4]])
            order_ind += 1
        orders_per_diff.append(cur_orders)
    return orders_per_diff

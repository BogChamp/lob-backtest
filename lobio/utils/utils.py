import numpy as np
import polars as pl
from typing import Sequence, Tuple
from tqdm import tqdm
from bisect import bisect_left

from lobio.lob.limit_order import Order, AMOUNT_TICK, EventType, OrderType, Side, TraderId
from lobio.lob.order_book import OrderBook, OrderBookSimple, TOP_N, OrderBookSimple2

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

def group_diffs(diffs_prepared: np.ndarray[int]) -> Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]]:
    diffs_df = pl.DataFrame(diffs_prepared, ('T', 'base', 'quote', 'side'))
    diffs_grouped_T = diffs_df.group_by('T').agg(pl.col('base'), pl.col('quote'), pl.col('side')).sort(by='T')

    diffs_grouped = []
    for diff in tqdm(diffs_grouped_T.iter_rows(), total=len(diffs_grouped_T)):
        diffs_grouped.append(get_grouped_diff(diff))
    
    return diffs_grouped

def get_initial_order_book(init_lob_data: np.ndarray[int], lob_class: OrderBook|OrderBookSimple|OrderBookSimple2):
    ts, init_lob = init_lob_data[-1][0], init_lob_data[:-1]
    bids = init_lob[:, :2]
    asks = init_lob[:, 2:]
    init_lob_dict = {"lastUpdateId": ts, "bids": bids, "asks": asks}
    order_book = lob_class.create_lob_init(init_lob_dict)
    return order_book

def check_if_diffs_sorted(diffs_grouped: Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]]):
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

def find_price_in_asks(asks: Sequence[Tuple[int, int]], price: int):
    price_level = []
    for ask in asks:
        if ask[0] == price:
            return (True, ask)
        elif ask[0] > price:
            return (False, price_level)
    return (False, price_level)

def find_price_in_bids(bids: Sequence[Tuple[int, int]], price: int):
    price_level = []
    for bid in bids:
        if bid[0] == price:
            return (True, bid)
        elif bid[0] < price:
            return (False, price_level)
    return (False, price_level)

def group_historical_trades(aggtrades_raw: np.ndarray[int], 
                            diffs_grouped: Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]]) \
                                -> Sequence[Sequence[Tuple[int, int, int, int]]]:
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
                                Tuple[list[Tuple[int, int, int, int, int, int, int]], list[Tuple[int, int, int, int, int, int, int]]]:
    ob = get_initial_order_book(init_lob, OrderBookSimple)
    #ob2 = get_initial_order_book(init_lob, OrderBookSimple)
    additional_data = [] # transaction time, order of event, diff number before which happened, p, q change, side, event type
    trades_prepared = []

    between_diffs_index = 0

    diffs_grouped_preprocess = [(init_lob[-1][0], [], [])] + diffs_grouped
    for i, diff in enumerate(tqdm(diffs_grouped_preprocess[1:])):
        cur_trades = trades_per_diff[i]
        cur_sells = [(diffs_grouped_preprocess[i][0], 0, None)] # ts, price, trade order
        cur_buys = [(diffs_grouped_preprocess[i][0], np.inf, None)]
        for n, trade in enumerate(cur_trades):
            bids = ob.bids
            asks = ob.asks
            if trade[3] == Side.SELL:
                unseen_ts = cur_sells[0][0] # earliest time for unseen change in ob
                if not len(bids) or trade[1] > bids[0][0]:
                    j = 0   # check if trade on opposite side's price
                    updates = []
                    while j < len(asks) and trade[1] >= asks[j][0]: # remove liquidity on asks, cause there was bid liquidity
                        #ts somewhere from diff_ts to now
                        additional_data.append([unseen_ts, between_diffs_index, i, asks[j][0], -asks[j][1], Side.SELL, EventType.DIFF])
                        between_diffs_index += 1
                        updates.append([asks[j][0], 0])  
                        j += 1
                    ob.update_asks(updates)

                    event_type = EventType.DIFF
                    for ts_price in cur_sells[::-1]: # find last ask trade which price is less that current - only since then bid liquidty may occure
                        if trade[1] > ts_price[1]:
                            unseen_ts = ts_price[0]
                            if ts_price[2] is not None:
                                event_type = EventType.LIMIT

                    #trade_since_liquidity_may_happened = cur_trades[last_ask_ind]
                    for trade_to_check in cur_trades[n:ts_price[2]:-1]: # check if buy trades was before and liquidity intersected
                        if trade_to_check[3] == Side.BUY and trade_to_check[1] <= trade[1]:
                            unseen_ts = trade_to_check[0]
                            event_type = EventType.LIMIT
                            break
                    #ts from last lower buy or sell to now
                    additional_data.append([unseen_ts, between_diffs_index, i, trade[1], trade[2], Side.BUY, event_type])
                    between_diffs_index += 1
                    ob.set_limit_order([trade[1], trade[2], Side.BUY])
                    
                elif trade[1] < bids[0][0]:
                    j = 0
                    updates = []
                    while j < len(bids) and trade[1] < bids[j][0]:
                        #ts somewhere from diff_ts to now
                        additional_data.append([unseen_ts, between_diffs_index, i, bids[j][0], -bids[j][1], Side.BUY, EventType.DIFF])
                        between_diffs_index += 1
                        updates.append([bids[j][0], 0])
                        j += 1
                    ob.update_bids(updates)

                    if len(bids) and trade[1] == bids[0][0]:
                        if trade[2] > bids[0][1]:
                            #ts somewhere from diff_ts to now
                            additional_data.append([unseen_ts, between_diffs_index, i, trade[1], trade[2]-bids[0][1], Side.BUY, EventType.DIFF])
                            between_diffs_index += 1
                            ob.set_limit_order([trade[1], trade[2]-bids[0][1], Side.BUY])
                    else:
                        #ts somewhere from diff_ts to now
                        additional_data.append([unseen_ts, between_diffs_index, i, trade[1], trade[2], Side.BUY, EventType.DIFF]) # MAKERS
                        between_diffs_index += 1
                        ob.set_limit_order([trade[1], trade[2], Side.BUY])
                elif trade[2] > bids[0][1]:
                    #ts somewhere from diff_ts to now
                    additional_data.append([unseen_ts, between_diffs_index, i, trade[1], trade[2]-bids[0][1], Side.BUY, EventType.DIFF]) # MAKERS
                    between_diffs_index += 1
                    ob.set_limit_order([trade[1], trade[2]-bids[0][1], Side.BUY])

                if trade[1] > cur_sells[-1][1]: # update sell trades info
                    cur_sells.append((trade[0], trade[1], n))
                elif trade[1] < cur_sells[-1][1]:
                    for ind, ts_price in enumerate(cur_sells[1:]):
                        if trade[1] < ts_price[1]:
                            cur_sells[ind+1] = (trade[0], trade[1], n)
                            cur_sells = cur_sells[:ind+2]
                            break
            else:
                unseen_ts = cur_buys[0][0]
                if not len(asks) or trade[1] < asks[0][0]:
                    j = 0   # check if trade on opposite side's price
                    updates = []
                    while j < len(bids) and trade[1] <= bids[j][0]: # remove liquidity on bids, cause there was ask liquidity
                        #ts somewhere from diff_ts to now
                        additional_data.append([unseen_ts, between_diffs_index, i, bids[j][0], -bids[j][1], Side.BUY, EventType.DIFF]) # add to diff
                        between_diffs_index += 1
                        updates.append([bids[j][0], 0])
                        j += 1
                    ob.update_bids(updates)

                    event_type = EventType.DIFF
                    for ts_price in cur_buys[::-1]: # find last bid trade which price is less that current - only since then ask liquidty may occure
                        if trade[1] < ts_price[1]:
                            unseen_ts = ts_price[0]
                            if ts_price[2] is not None:
                                event_type = EventType.LIMIT

                    for trade_to_check in cur_trades[n:ts_price[2]:-1]: # check if sell trades was before and liquidity intersected
                        if trade_to_check[3] == Side.SELL and trade_to_check[1] >= trade[1]:
                            unseen_ts = trade_to_check[0]
                            event_type = EventType.LIMIT
                            break
                    # ts is somewhere from last higher sell or buy to now
                    additional_data.append([unseen_ts, between_diffs_index, i, trade[1], trade[2], Side.SELL, event_type])
                    ob.set_limit_order([trade[1], trade[2], Side.SELL])
                    between_diffs_index += 1
                elif trade[1] > asks[0][0]:
                    j = 0
                    updates = []
                    while j < len(asks) and trade[1] > asks[j][0]:
                        #ts somewhere from diff_ts to now
                        additional_data.append([unseen_ts, between_diffs_index, i, asks[j][0], -asks[j][1], Side.SELL, EventType.DIFF]) 
                        between_diffs_index += 1
                        updates.append([asks[j][0], 0])
                        j += 1
                    ob.update_asks(updates)

                    if len(asks) and trade[1] == asks[0][0]:
                        if trade[2] > asks[0][1]:
                            #ts somewhere from diff_ts to now
                            additional_data.append([unseen_ts, between_diffs_index, i, trade[1], trade[2]-asks[0][1], Side.SELL, EventType.DIFF])
                            between_diffs_index += 1
                            ob.set_limit_order([trade[1], trade[2]-asks[0][1], Side.SELL])
                    else:
                        #ts somewhere from diff_ts to now
                        additional_data.append([unseen_ts, between_diffs_index, i, trade[1], trade[2], Side.SELL, EventType.DIFF]) # MAKERS
                        between_diffs_index += 1
                        ob.set_limit_order([trade[1], trade[2], Side.SELL])
                elif trade[2] > asks[0][1]:
                    #ts somewhere from diff_ts to now
                    additional_data.append([unseen_ts, between_diffs_index, i, trade[1], trade[2]-asks[0][1], Side.SELL, EventType.DIFF]) # MAKERS
                    between_diffs_index += 1
                    ob.set_limit_order([trade[1], trade[2]-asks[0][1], Side.SELL])

                if trade[1] < cur_buys[-1][1]:
                    cur_buys.append((trade[0], trade[1], n))
                elif trade[1] > cur_buys[-1][1]:
                    for ind, ts_price in enumerate(cur_buys[1:]):
                        if trade[1] > ts_price[1]:
                            cur_buys[ind+1] = (trade[0], trade[1], n)
                            cur_buys = cur_buys[:ind+2]
                            break
            #ts is given
            trades_prepared.append([trade[0], between_diffs_index, i, trade[1], trade[2], trade[3], EventType.MARKET])
            between_diffs_index += 1
            ob.set_market_order([trade[2], trade[3]])
            
        ob.apply_historical_update(diff)
        #ob2.apply_historical_update(diff)
        #assert ob == ob2
    return trades_prepared, additional_data

def split_unseen_dynamic(unseen_dynamic: Sequence[Tuple[int, int, int, int, int, int, int]]) -> Tuple[list[Tuple[int, int, int, int]], 
                                                                                                      list[Tuple[int, int, int, int, int, int, int]], 
                                                                                                      list[Tuple[int, int, int]]]:
    diffs = []
    orders = []
    initial_state_update = []

    for event in tqdm(unseen_dynamic):
        if event[6] == EventType.DIFF:
            if event[2] == 0:
                initial_state_update.append([event[3], event[4], event[5]])
            else:
                diffs.append([event[2], event[3], event[4], event[5]])
        else:
            orders.append(event)
    
    return diffs, orders, initial_state_update

def update_diffs(unseen_diffs: list[Tuple[int, int, int, int]], init_lob: np.ndarray[int],
                 diffs_grouped: list[Tuple[int, list[Tuple[int, int]], list[Tuple[int, int]]]]) -> \
                    list[Tuple[int, list[Tuple[int, int]], list[Tuple[int, int]]]]:
    final_diffs = [] # part of functionality can be passed to obsimple class
    unseen_diff_ind = 0
    ob = get_initial_order_book(init_lob, OrderBookSimple)
    for i, diff in enumerate(tqdm(diffs_grouped)):
        new_diff = ob.track_diff(diff)

        while unseen_diff_ind < len(unseen_diffs) and (unseen_diffs[unseen_diff_ind][0] - 1) == i:
            if unseen_diffs[unseen_diff_ind][3] == Side.BUY:
                index = bisect_left(
                    ob.bids,  -unseen_diffs[unseen_diff_ind][1], key=lambda x: -x[0]
                )
                if index > len(ob.bids) or ob.bids[index][0] != unseen_diffs[unseen_diff_ind][1]:
                    change_bid_diff(new_diff[1], [unseen_diffs[unseen_diff_ind][1], unseen_diffs[unseen_diff_ind][2]])
                else:
                    new_amount = ob.bids[index][1] + unseen_diffs[unseen_diff_ind][2]
                    change_bid_diff(new_diff[1], [unseen_diffs[unseen_diff_ind][1], new_amount])
            else:
                index = bisect_left(
                    ob.asks,  unseen_diffs[unseen_diff_ind][1], key=lambda x: x[0]
                )
                if index > len(ob.asks) or ob.asks[index][0] != unseen_diffs[unseen_diff_ind][1]:
                    change_ask_diff(new_diff[2], [unseen_diffs[unseen_diff_ind][1], unseen_diffs[unseen_diff_ind][2]])
                else:
                    new_amount = ob.asks[index][1] + unseen_diffs[unseen_diff_ind][2]
                    change_ask_diff(new_diff[2], [unseen_diffs[unseen_diff_ind][1], new_amount])
    
            unseen_diff_ind += 1

        final_diffs.append(new_diff)
    
    return final_diffs

def collect_new_bid_diff(top_bids_before: Sequence[Tuple[int, int]], 
                         top_bids_after: Sequence[Tuple[int, int]], 
                         diff_bids: Sequence[Tuple[int, int]]) -> list[Tuple[int, int]]:
    ind_before = 0
    ind_after = 0
    ind_diff = 0
    result = []

    while ind_before < len(top_bids_before) and ind_after < len(top_bids_after):
        if top_bids_before[ind_before][0] > top_bids_after[ind_after][0]:
            result.append((top_bids_before[ind_before][0], 0))
            # check
            # while diff[1][ind_diff][0] > top_bids_before[ind_before][0]:
            #     ind_diff += 1
            
            # assert diff[1][ind_diff][0] == top_bids_before[ind_before][0]
            # assert diff[1][ind_diff][1] == 0
            ind_before += 1
        elif top_bids_before[ind_before][0] < top_bids_after[ind_after][0]:
            result.append(top_bids_after[ind_after])
            # check
            # while diff[1][ind_diff][0] > top_bids_after[ind_after][0]:
            #     ind_diff += 1
            
            # assert diff[1][ind_diff][0] == top_bids_after[ind_after][0]
            # assert diff[1][ind_diff][1] == top_bids_after[ind_after][1]
            ind_after += 1
        else: # due to empty diffs changed!
            while ind_diff < len(diff_bids) and diff_bids[ind_diff][0] > top_bids_after[ind_after][0]:
                ind_diff += 1
            
            if len(result) == 0 or (ind_diff != len(diff_bids) and diff_bids[ind_diff][0] == top_bids_after[ind_after][0]):
                result.append(top_bids_after[ind_after])
                #assert top_bids_after[ind_after][1] == diff[1][ind_diff][1]
            #else:
                #assert top_bids_before[ind_before][1] == top_bids_after[ind_after][1]
            ind_before += 1
            ind_after += 1
    
    while ind_after < len(top_bids_after):
        result.append(top_bids_after[ind_after])
        # while ind_diff < len(diff[1]) and diff[1][ind_diff][0] > top_bids_after[ind_after][0]:
        #         ind_diff += 1
            
        # if ind_diff == len(diff[1]) or diff[1][ind_diff][0] != top_bids_after[ind_after][0]:
        #     assert top_bids_after[ind_after][0] < top_bids_before[-1][0]
        # else:
        #     assert diff[1][ind_diff][0] == top_bids_after[ind_after][0]
        #     assert diff[1][ind_diff][1] == top_bids_after[ind_after][1]
        ind_after += 1
    
    return result

def collect_new_ask_diff(top_asks_before: Sequence[Tuple[int, int]], 
                         top_asks_after: Sequence[Tuple[int, int]], 
                         diff_asks: Sequence[Tuple[int, int]]) -> list[Tuple[int, int]]:
    ind_before = 0
    ind_after = 0
    ind_diff = 0
    result = []

    while ind_before < len(top_asks_before) and ind_after < len(top_asks_after):
        if top_asks_before[ind_before][0] < top_asks_after[ind_after][0]:
            result.append((top_asks_before[ind_before][0], 0))
            # check
            # while diff[2][ind_diff][0] < top_asks_before[ind_before][0]:
            #     ind_diff += 1
            
            # assert diff[2][ind_diff][0] == top_asks_before[ind_before][0]
            # assert diff[2][ind_diff][1] == 0 
            ind_before += 1
        elif top_asks_before[ind_before][0] > top_asks_after[ind_after][0]:
            result.append(top_asks_after[ind_after])
            # check
            # while diff[2][ind_diff][0] < top_asks_after[ind_after][0]:
            #     ind_diff += 1

            # assert diff[2][ind_diff][0] == top_asks_after[ind_after][0]
            # assert diff[2][ind_diff][1] == top_asks_after[ind_after][1]
            ind_after += 1
        else:
            while ind_diff < len(diff_asks) and diff_asks[ind_diff][0] < top_asks_after[ind_after][0]:
                ind_diff += 1
            
            if len(result) == 0 or (ind_diff != len(diff_asks) and diff_asks[ind_diff][0] == top_asks_after[ind_after][0]):
                result.append(top_asks_after[ind_after])
                #assert top_asks_after[ind_after][1] == diff[2][ind_diff][1]
            #else:
                #assert top_asks_before[ind_before][1] == top_asks_after[ind_after][1]
            ind_before += 1
            ind_after += 1

    while ind_after < len(top_asks_after):
        result.append(top_asks_after[ind_after])
        # while ind_diff < len(diff[2]) and diff[2][ind_diff][0] < top_asks_after[ind_after][0]:
        #     ind_diff += 1

        # if ind_diff == len(diff[2]) or diff[2][ind_diff][0] != top_asks_after[ind_after][0]:
        #     assert top_asks_after[ind_after][0] > top_asks_before[-1][0]
        # else:
        #     assert diff[2][ind_diff][0] == top_asks_after[ind_after][0]
        #     assert diff[2][ind_diff][1] == top_asks_after[ind_after][1]
        ind_after += 1
    return result

def cut_diffs(init_lob: np.ndarray[int],
              new_diffs: Sequence[Tuple[int, list[Tuple[int, int]], list[Tuple[int, int]]]],
              orders_per_diff: Sequence[Sequence[Tuple[int, Order]]]) -> Sequence[Tuple[int, list[Tuple[int, int]], list[Tuple[int, int]]]]:
    ob = get_initial_order_book(init_lob, OrderBookSimple)
    diffs_cut = []
    for i, diff in enumerate(tqdm(new_diffs)):
        top_bids_before = [(ob.bids[i][0], ob.bids[i][1]) for i in range(TOP_N)]
        top_asks_before = [(ob.asks[i][0], ob.asks[i][1]) for i in range(TOP_N)]

        cur_orders = orders_per_diff[i]
        for _, order in cur_orders:
            if order.type == OrderType.MARKET:
                ob.set_market_order([order.quote, order.side])
            else:
                ob.set_limit_order([order.base, order.quote, order.side])

        ob.apply_historical_update(diff)

        top_bids_after = [(ob.bids[i][0], ob.bids[i][1]) for i in range(TOP_N)]
        top_asks_after = [(ob.asks[i][0], ob.asks[i][1]) for i in range(TOP_N)]
        
        cur_bid_cut = collect_new_bid_diff(top_bids_before, top_bids_after, diff[1])
        cur_ask_cut = collect_new_ask_diff(top_asks_before, top_asks_after, diff[2])
        diffs_cut.append((diff[0], cur_bid_cut, cur_ask_cut))
    
    return diffs_cut

def prepare_init_lob(init_lob: np.ndarray[int],
                     initial_state_update: Sequence[Tuple[int, int, int]]) -> np.ndarray[int]:
    ob = get_initial_order_book(init_lob, OrderBookSimple)
    for diff_change in tqdm(initial_state_update):
        if diff_change[2] == Side.BUY:
            index = bisect_left(ob.bids,  -diff_change[0], key=lambda x: -x[0])
            if index == len(ob.bids):
                if diff_change[1] > 0:
                    ob.bids.append([diff_change[0], diff_change[1]])
            elif ob.bids[index][0] == diff_change[0]:
                ob.bids[index][1] += diff_change[1]
                if ob.bids[index][1] == 0:
                    del ob.bids[index]
            else:
                if diff_change[1] > 0:
                    ob.bids.insert(index, [diff_change[0], diff_change[1]])
        else:
            index = bisect_left(ob.asks,  diff_change[0], key=lambda x: x[0])
            if index == len(ob.asks):
                if diff_change[1] > 0:
                    ob.asks.append([diff_change[0], diff_change[1]])
            elif ob.asks[index][0] == diff_change[0]:
                ob.asks[index][1] += diff_change[1]
                if ob.asks[index][1] == 0:
                    del ob.asks[index]
            else:
                if diff_change[1] > 0:
                    ob.asks.insert(index, [diff_change[0], diff_change[1]])
    
    min_len = min(len(ob.bids), len(ob.asks))
    bids_prepared = ob.bids[:min_len]
    asks_prepared = ob.asks[:min_len]
    init_lob_data = np.concatenate([bids_prepared, asks_prepared], axis=1)
    init_lob_data = np.round(init_lob_data)
    init_lob_data = np.append(init_lob_data, [[init_lob[-1][0]] * init_lob_data.shape[1]], axis=0).astype(int)

    return init_lob_data

def merge_orders(sorted_list1: Sequence[Tuple[int, int, int, int, int, int, int]], 
                 sorted_list2: Sequence[Tuple[int, int, int, int, int, int, int]]) -> Sequence[Tuple[int, int, int, int, int, int]]:
    ind1 = 0
    ind2 = 0

    merged_array = [[0, 0, 0, 0, 0, 0]] * (len(sorted_list1) + len(sorted_list2)) 
    # (ts, diff num, p, q, side, type)
    while ind1 < len(sorted_list1) and ind2 < len(sorted_list2):
        if sorted_list1[ind1][0] < sorted_list2[ind2][0]:
            order_type = OrderType.MARKET if sorted_list1[ind1][6] == EventType.MARKET else OrderType.LIMIT
            merged_array[ind1+ind2] = [sorted_list1[ind1][0], sorted_list1[ind1][2], sorted_list1[ind1][3], 
                                       sorted_list1[ind1][4], sorted_list1[ind1][5], order_type]
            ind1 += 1
        elif sorted_list1[ind1][0] > sorted_list2[ind2][0]:
            order_type = OrderType.MARKET if sorted_list2[ind2][6] == EventType.MARKET else OrderType.LIMIT
            merged_array[ind1+ind2] = [sorted_list2[ind2][0], sorted_list2[ind2][2], sorted_list2[ind2][3], 
                                       sorted_list2[ind2][4], sorted_list2[ind2][5], order_type]
            ind2 += 1
        elif sorted_list1[ind1][1] < sorted_list2[ind2][1]:
            order_type = OrderType.MARKET if sorted_list1[ind1][6] == EventType.MARKET else OrderType.LIMIT
            merged_array[ind1+ind2] = [sorted_list1[ind1][0], sorted_list1[ind1][2], sorted_list1[ind1][3], 
                                       sorted_list1[ind1][4], sorted_list1[ind1][5], order_type]
            ind1 += 1
        else:
            order_type = OrderType.MARKET if sorted_list2[ind2][6] == EventType.MARKET else OrderType.LIMIT
            merged_array[ind1+ind2] = [sorted_list2[ind2][0], sorted_list2[ind2][2], sorted_list2[ind2][3], 
                                       sorted_list2[ind2][4], sorted_list2[ind2][5], order_type]
            ind2 += 1

    while ind1 < len(sorted_list1):
        order_type = OrderType.MARKET if sorted_list1[ind1][6] == EventType.MARKET else OrderType.LIMIT
        merged_array[ind1+ind2] = [sorted_list1[ind1][0], sorted_list1[ind1][2], sorted_list1[ind1][3], 
                                    sorted_list1[ind1][4], sorted_list1[ind1][5], order_type]
        ind1 += 1
    
    while ind2 < len(sorted_list2):
        order_type = OrderType.MARKET if sorted_list2[ind2][6] == EventType.MARKET else OrderType.LIMIT
        merged_array[ind1+ind2] = [sorted_list2[ind2][0], sorted_list2[ind2][2], sorted_list2[ind2][3], 
                                   sorted_list2[ind2][4], sorted_list2[ind2][5], order_type]
        ind2 += 1
    
    return merged_array

def aggregate_orders(orders_flow: Sequence[Tuple[int, int, int, int, int, int]]) -> np.ndarray[int]:
    aggregated_orders = [orders_flow[0]]
    for order in tqdm(orders_flow[1:]):
        if order[0] == aggregated_orders[-1][0] and order[1] == aggregated_orders[-1][1] and \
            order[4] == aggregated_orders[-1][4] and order[5] == aggregated_orders[-1][5]:
            if order[5] == OrderType.MARKET:
                aggregated_orders[-1][3] += order[3] # AGGREGATE market orders by volume, price unimportant
            elif order[2] == aggregated_orders[-1][2]:
                aggregated_orders[-1][3] += order[3]  # AGGREGATE limit orders by volume if price levels equal
            else:
                aggregated_orders.append(order)
        else:
            aggregated_orders.append(order)
    return np.array(aggregated_orders)

def prepare_diffs(diffs_transformed: Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]]) -> np.ndarray[int]:
    diff_sequence = []
    for diff in tqdm(diffs_transformed):
        ts = diff[0]
        bids = diff[1]
        asks = diff[2]

        for bid in bids:
            diff_sequence.append((ts, bid[0], bid[1], Side.BUY))

        for ask in asks:
            diff_sequence.append((ts, ask[0], ask[1], Side.SELL))

    # diffs_prepared = pl.DataFrame(diff_sequence, ('T', 'base', 'quote', 'side'))
    # diffs_prepared = diffs_prepared.with_columns(pl.col('base').cast(pl.UInt32), 
    #                     pl.col('quote').cast(pl.Int64), 
    #                     pl.col('side').cast(pl.Int8),
    #                     pl.col('T').cast(pl.UInt64))
    diffs_prepared = np.array(diff_sequence)
    return diffs_prepared

def change_bid_diff(bid_diff: list[Tuple[int, int]], change: Tuple[int, int]):
    index = bisect_left(bid_diff,  -change[0], key=lambda x: -x[0])
    if index == len(bid_diff):
        bid_diff.append(change)
    elif bid_diff[index][0] == change[0]:
        bid_diff[index][1] = change[1]
    else:
        bid_diff.insert(index, change)

def change_ask_diff(ask_diff: list[Tuple[int, int]], change: Tuple[int, int]):
    index = bisect_left(ask_diff,  change[0], key=lambda x: x[0])
    if index == len(ask_diff):
        ask_diff.append(change)
    elif ask_diff[index][0] == change[0]:
        ask_diff[index][1] = change[1]
    else:
        ask_diff.insert(index, change)

def group_orders(orders_prepared_np: np.ndarray[int], diffs_total: int) -> Sequence[Sequence[Tuple[int, Order]]]:
    orders_per_diff = []
    order_ind = 0
    for i in tqdm(range(diffs_total)):
        cur_orders = []
        while order_ind < len(orders_prepared_np) and orders_prepared_np[order_ind][1] == i:
            cur_orders.append((orders_prepared_np[order_ind][0], 
                               Order(orders_prepared_np[order_ind][2], orders_prepared_np[order_ind][3], 
                                     orders_prepared_np[order_ind][4], orders_prepared_np[order_ind][5], TraderId.MARKET)))
            order_ind += 1
        orders_per_diff.append(cur_orders)
    return orders_per_diff

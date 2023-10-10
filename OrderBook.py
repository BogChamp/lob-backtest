from typing import Sequence
from collections import defaultdict
from bisect import bisect_left

from LimitOrder import LimitOrder, PRICE_TICK, AMOUNT_TICK
from PriceLevel import PriceLevel

BUY: int = 0
SELL: int = 1

MARKET_ID: int = 0
MM_ID: int = 1


class OrderBookBase:
    def __init__(self, bids: Sequence[PriceLevel] = [], asks: Sequence[PriceLevel] = []):
        self.bids = sorted(bids, reverse=True, key=lambda x: x.price)
        self.asks = sorted(asks, key=lambda x: x.price)
    
    def __repr__(self):
        ob_repr = ''

        min_len = min(len(self.bids), len(self.asks))
        for i in range(min_len):
            bid_str = repr(self.bids[i])
            ob_repr += bid_str + (36 - len(bid_str)) * ' ' + repr(self.asks[i]) + '\n'
        
        if len(self.bids) > min_len:
            remain_price_levels = self.bids[min_len:]
            indent = 0
        else:
            remain_price_levels = self.asks[min_len:]
            indent = 36
        
        for p_l in remain_price_levels:
            ob_repr += indent * ' ' + repr(p_l) + '\n'
        
        return ob_repr
    
    def get_state(self):
        return self.bids, self.asks
    
    def get_bids(self):
        return self.bids
    
    def get_asks(self):
        return self.asks
    
    def bid_price(self):
        return self.bids[0].price

    def ask_price(self):
        return self.asks[0].price

    def mid_price(self):
        return (self.bids[0].price + self.asks[0].price) / 2

    def bid_ask_spread(self):
        return self.asks[0].price - self.bids[0].price

    def market_depth(self):
        return self.asks[-1].price - self.bids[-1].price
    
    @staticmethod
    def matching_engine(price_levels: Sequence[PriceLevel], 
                        limit_order: LimitOrder):
        matches_info = defaultdict(list)
        remain_amount = limit_order.amount
        sign = 2 * limit_order.side - 1

        for i, price_level in enumerate(price_levels):
            if (sign * price_level.price < sign * limit_order.price) or (remain_amount == 0):
                break
            this_price = price_level.price
            remain_amount, cur_match_info = price_level.execute_limit_order(remain_amount)
            for k, v in cur_match_info.items():
                trader_info = matches_info[k]
                if len(trader_info):
                    trader_info[0] += v
                    trader_info[1] += v * this_price
                else:
                    trader_info = [v, v * this_price]
                matches_info[k] = trader_info

        return matches_info, remain_amount, i

    def set_limit_order(self, limit_order: LimitOrder):
        sign = 2 * limit_order.side - 1
        if sign == -1:
            price_levels = self.bids
            opposite_price_levels = self.asks
        elif sign == 1:
            price_levels = self.asks
            opposite_price_levels = self.bids
        else:
            raise Exception('WRONG SIDE')

        old_amount = limit_order.amount
        matches_info, remain_amount, p_l_eaten = self.matching_engine(opposite_price_levels, limit_order)

        if sign == -1:
            self.asks = self.asks[p_l_eaten:]
        else:
            self.bids = self.bids[p_l_eaten:]

        if remain_amount > 0:
            if remain_amount != old_amount:
                new_limit_order = LimitOrder(limit_order.price, remain_amount, 
                                          limit_order.side, limit_order.trader_id)
                price_levels.insert(0, PriceLevel(new_limit_order))
            else:
                index = bisect_left(price_levels, sign * limit_order.price, key=lambda x: sign * x.price)
                if index == len(price_levels):
                    price_levels.append(PriceLevel(limit_order))
                elif price_levels[index].price == limit_order.price:
                    price_levels[index].add_limit_order(limit_order)
                else:
                    price_levels.insert(index, PriceLevel(limit_order))

        return matches_info

    @staticmethod
    def create_lob_init(lob_state: dict):
        bids_raw = lob_state['bids']
        asks_raw = lob_state['asks']

        bids = []
        asks = []
        for bid_raw in bids_raw:
            bid = PriceLevel(LimitOrder(bid_raw[0], bid_raw[1], BUY, MARKET_ID))
            bids.append(bid)
        
        for ask_raw in asks_raw:
            ask = PriceLevel(LimitOrder(ask_raw[0], ask_raw[1], SELL, MARKET_ID))
            asks.append(ask)
        
        return OrderBookBase(bids, asks)


class OrderBookPrep(OrderBookBase):
    def track_diff_side(self, update, side: int):
        diff_new = []
        sign = 2 * side - 1
        if sign == -1:
            price_levels = self.bids
        elif sign == 1:
            price_levels = self.asks
        else:
            raise Exception('WRONG SIDE')
        
        for price_level in update:
            price = round(price_level[0], PRICE_TICK)
            amount = round(price_level[1], AMOUNT_TICK)
            index = bisect_left(price_levels, sign * price, key=lambda x: sign * x.price)
            if index == len(price_levels):
                if amount > 0:
                    price_levels.append(PriceLevel(LimitOrder(price, amount, side, MARKET_ID)))
                    diff_new.append([price, amount])
            elif price_levels[index].price == price:
                amount_change = round(amount - price_levels[index].amount, AMOUNT_TICK)
                if amount_change != 0:
                    diff_new.append([price, amount_change])
                    price_levels[index].change_liquidity(amount_change, MARKET_ID)
                    if price_levels[index].amount == 0: del price_levels[index]
            else:
                if amount > 0:
                    price_levels.insert(index, PriceLevel(LimitOrder(price, amount, side, MARKET_ID)))
                    diff_new.append([price, amount])
        return diff_new

    def track_diff(self, diff):
        bids_update = diff[1]
        asks_update = diff[2]

        bids_diff_new = self.track_diff_side(bids_update, BUY)
        asks_diff_new = self.track_diff_side(asks_update, SELL)
        
        return (diff[0], bids_diff_new, asks_diff_new)
    
    @staticmethod
    def create_lob_init(lob_state: dict):
        bids_raw = lob_state['bids']
        asks_raw = lob_state['asks']

        bids = []
        asks = []
        for bid_raw in bids_raw:
            bid = PriceLevel(LimitOrder(bid_raw[0], bid_raw[1], BUY, MARKET_ID))
            bids.append(bid)
        
        for ask_raw in asks_raw:
            ask = PriceLevel(LimitOrder(ask_raw[0], ask_raw[1], SELL, MARKET_ID))
            asks.append(ask)
        
        return OrderBookPrep(bids, asks)


class OrderBook(OrderBookBase):
    def update_limit_orders(self, updates, side):
        sign = 2 * side - 1
        if side == BUY:
            price_levels = self.bids
        elif side == SELL:
            price_levels = self.asks
        else:
            raise Exception("Wrong side!")
        
        for update in updates: # THINK ABOUT ORDER OF PRICE CHANGES
            index = bisect_left(price_levels, sign*update[0], key=lambda x: sign*x.price)
            if index == len(price_levels):
                if update[1] > 0:
                    self.set_limit_order(LimitOrder(update[0], update[1], side, MARKET_ID))
            elif price_levels[index].price == update[0]:
                price_levels[index].change_liquidity(update[1], MARKET_ID)
                if price_levels[index].amount == 0:
                    del price_levels[index]
            else:
                if update[1] > 0:
                    self.set_limit_order(LimitOrder(update[0], update[1], BUY, MARKET_ID))
            
    def apply_historical_update(self, updates):
        bids_update = updates[1]
        asks_update = updates[2]

        self.update_limit_orders(bids_update, BUY)
        self.update_limit_orders(asks_update, SELL)

    @staticmethod
    def create_lob_init(lob_state: dict):
        bids_raw = lob_state['bids']
        asks_raw = lob_state['asks']

        bids = []
        asks = []
        for bid_raw in bids_raw:
            bid = PriceLevel(LimitOrder(bid_raw[0], bid_raw[1], BUY, MARKET_ID))
            bids.append(bid)
        
        for ask_raw in asks_raw:
            ask = PriceLevel(LimitOrder(ask_raw[0], ask_raw[1], SELL, MARKET_ID))
            asks.append(ask)
        
        return OrderBook(bids, asks)

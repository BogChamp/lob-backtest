from typing import Sequence
from collections import defaultdict
from bisect import bisect_left

from LimitOrder import LimitOrder, PRICE_TICK, AMOUNT_TICK
from PriceLevel import PriceLevel

BUY: int = 0
SELL: int = 1

MARKET_ID: int = 0
MM_ID: int = 1

class MatchingEngine:
    def __init__(self):
        pass 
    
    @staticmethod
    def match_orders(price_levels: Sequence[PriceLevel], amount: float):
        remain_amount = amount
        price: float = 0.0
        remain_orders = []
        matches_info = defaultdict(list) # trader_info - [amount, price]

        for i, price_level in enumerate(price_levels):
            this_price: float = price_level.price
            this_amount: float = price_level.amount
            price += this_price * min(remain_amount, this_amount)
            remain_amount, this_match_info = price_level.execute_limit_order(remain_amount)

            for k, v in this_match_info.items():
                trader_info = matches_info[k]
                if len(trader_info):
                    trader_info[0] += v
                    trader_info[1] += v * this_price
                else:
                    trader_info = [v, v * this_price]
                matches_info[k] = trader_info

            if not remain_amount:
                if price_level.amount:
                    remain_orders += [price_level]
                remain_orders += price_levels[i+1:]
                break

        return matches_info, remain_orders, remain_amount

class OrderBookBase:
    def __init__(self, bids: Sequence[PriceLevel] = [], asks: Sequence[PriceLevel] = [], 
                 matching_engine: MatchingEngine = MatchingEngine()):
        self.bids = sorted(bids, reverse=True, key=lambda x: x.price)
        self.asks = sorted(asks, key=lambda x: x.price)
        self.matching_engine = matching_engine
    
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
    
    def set_ask_order(self, limit_order: LimitOrder):
        index = len(self.bids)
        for i, price_level in enumerate(self.bids):
            if price_level.price < limit_order.price:
                index = i
                break

        eligible_bids = self.bids[:index]
        ineligible_bids = self.bids[index:]

        matches_info, remain_bids, remain_amount = self.matching_engine.match_orders(eligible_bids, limit_order.amount)
        new_bids: Sequence[PriceLevel] = remain_bids + ineligible_bids
        new_asks: Sequence[PriceLevel] = self.asks

        if remain_amount > 0:
            new_limit_order = LimitOrder(limit_order.price, remain_amount, 
                                         limit_order.side, limit_order.trader_id)

            index = len(new_asks)
            for i, price_level in enumerate(new_asks):
                if price_level.price >= limit_order.price:
                    index = i
                    break

            if index == len(new_asks):
                new_asks.append(PriceLevel(new_limit_order))
            elif new_asks[index].price != limit_order.price:
                new_asks.insert(index, PriceLevel(new_limit_order))
            else:
                new_asks[index].add_limit_order(new_limit_order)
        
        self.bids = new_bids
        self.asks = new_asks

        return matches_info

    def set_bid_order(self, limit_order: LimitOrder):
        index = len(self.asks)
        for i, price_level in enumerate(self.asks):
            if price_level.price > limit_order.price:
                index = i
                break
 
        eligible_asks = self.asks[:index]
        ineligible_asks = self.asks[index:]

        matches_info, remain_asks, remain_amount = self.matching_engine.match_orders(eligible_asks, limit_order.amount)
        new_asks: Sequence[PriceLevel] = remain_asks + ineligible_asks
        new_bids: Sequence[PriceLevel] = self.bids

        if remain_amount > 0:
            new_limit_order = LimitOrder(limit_order.price, remain_amount, 
                                         limit_order.side, limit_order.trader_id)

            index = len(new_bids)
            for i, price_level in enumerate(new_bids):
                if price_level.price <= limit_order.price:
                    index = i
                    break
            
            if index == len(new_bids):
                new_bids.append(PriceLevel(new_limit_order))
            elif new_bids[index].price != limit_order.price:
                new_bids.insert(index, PriceLevel(new_limit_order))
            else:
                new_bids[index].add_limit_order(new_limit_order)
        
        self.bids = new_bids
        self.asks = new_asks

        return matches_info

    def set_order(self, limit_order: LimitOrder):
        if limit_order.side == SELL:
            matches_info = self.set_ask_order(limit_order)
        elif limit_order.side == BUY:
            matches_info = self.set_bid_order(limit_order)
        else:
            raise Exception("WRONG SIDE!")
        
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
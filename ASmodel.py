
import numpy as np
from OrderBook import OrderBookBase, MM_ID, BUY, SELL
from LimitOrder import LimitOrder, PRICE_TICK

class AvellanedaStoikov:
    def __init__(self, T: float, t_start: float, q0: float = 0, k: float = 1.5, 
                 sigma: float = 2, gamma: float = 0.1, q_max: float = 10**5):
        self.T = T
        self.t_start = t_start
        self.q0 = q0
        self.k = k
        self.sigma = sigma
        self.gamma = gamma
        self.q_max = q_max 

        self.q = self.q0

    def get_indifference_price(self, mid_price: float, timestamp):
        if timestamp > self.T:
            raise Exception("Time for trading is over")
        
        r = mid_price - self.q * self.gamma * self.sigma**2 * (self.T - timestamp) / (self.T - self.t_start)
        return r
    
    def get_optimal_spread(self, timestamp):
        if timestamp > self.T:
            raise Exception("Time for trading is over")
        
        optimal_spread = self.gamma * self.sigma**2 * (self.T - timestamp) / (self.T - self.t_start) + \
                         2 / self.gamma * np.log1p(self.gamma / self.k)

        return optimal_spread
        
    def update_inventory(self, q: float):
        self.q = q
    
    def get_bid_ask_price(self, lob_state: OrderBookBase, timestamp):
        r = self.get_indifference_price(lob_state.mid_price(), timestamp)
        optimal_spread = self.get_optimal_spread(timestamp)

        bid_price = r - optimal_spread / 2
        ask_price = r + optimal_spread / 2

        bid_price = round(bid_price, PRICE_TICK)
        ask_price = round(ask_price, PRICE_TICK)
    
        return bid_price, ask_price
    
    def bid_ask_limit_orders(self, lob_state: OrderBookBase, timestamp, q = None):
        if q is not None:
            self.update_inventory(q)
        bid_price, ask_price = self.get_bid_ask_price(lob_state, timestamp)

        bid_order = LimitOrder(bid_price, self.q_max, BUY, MM_ID)
        ask_order = LimitOrder(ask_price, self.q_max, SELL, MM_ID)

        return bid_order, ask_order
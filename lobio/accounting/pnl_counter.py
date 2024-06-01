import numpy as np


class PnL_Counter:
    """Class for pnl counting."""

    def __init__(self, q0: int = 0):
        self.q0 = q0
        self.quote_history = [q0]
        self.price_history = []

    def collect_statistic(self, new_q: int, new_price: float):
        self.quote_history.append(new_q)
        self.price_history.append(new_price)
    
    def calculate_pnl(self) -> int:
        pnl = []
        for i, price in enumerate(self.price_history):
            pnl.append(price * (self.quote_history[i+1] - self.quote_history[i]))
        return np.cumsum(pnl)

    def reset(self):
        self.quote_history = [self.q0]
        self.price_history = []

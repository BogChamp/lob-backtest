import numpy as np
from lobio.lob.limit_order import PRICE_TICK, AMOUNT_TICK

class PnL_Counter:
    """Class for pnl counting."""

    def __init__(self, q0: int = 0):
        self.q0 = q0
        self.quote_history = [q0]
        self.price_history = []

    def collect_statistic(self, new_q: int, new_price: float):
        self.quote_history.append(new_q)
        self.price_history.append(new_price)
    
    def calculate_pnl(self) -> list[int]:
        pnl = []
        for i, price in enumerate(self.price_history):
            pnl.append(price * (self.quote_history[i+1] - self.quote_history[i]))
        return np.cumsum(pnl)

    def reset(self):
        self.quote_history = [self.q0]
        self.price_history = []

class PnL_Counter2:
    """Class for pnl counting."""

    def __init__(self, q0: int = 0):
        self.q0 = q0
        self.quote_history = [q0]
        self.price_history = []
        self.base_history = []

    def collect_statistic(self, new_q: int, new_price: float, new_base: float):
        self.quote_history.append(new_q / 10**(AMOUNT_TICK))
        self.price_history.append(new_price / 10**(PRICE_TICK))
        self.base_history.append(new_base / 10**(PRICE_TICK + AMOUNT_TICK))
    
    def calculate_pnl(self) -> list[int]:
        pnl = []
        for i, base in enumerate(self.base_history):
            pnl.append(base + self.price_history[i] * self.quote_history[i])
        return np.array(pnl)

    def reset(self):
        self.quote_history = [self.q0]
        self.price_history = []
        self.base_history = []

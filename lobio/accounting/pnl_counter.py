import numpy as np


class PnL_Counter:
    """Class for pnl counting."""

    def __init__(self, q0: int = 0):
        self.q0 = q0
        self.quote_history = [q0]
        self.price_history = []
        # self.asset_purchased = 0.0
        # self.money_spent = 0.0

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
    

    # def update(self, asset_purchased, money_spent):
    #     self.asset_purchased += asset_purchased
    #     self.money_spent += money_spent

    # def unrealized_pnl(self, cur_amount, ask_price):
    #     if self.asset_purchased > 0:
    #         self.pnl += cur_amount * (ask_price - self.money_spent / self.asset_purchased)

    # def realized_pnl(self, amount_sold, price):
    #     if self.asset_purchased > 0:
    #         self.pnl += price - amount_sold * (self.money_spent / self.asset_purchased)

class PnL_Counter:
    def __init__(self):
        self.pnl = 0.0
        self.asset_purchased = 0.0
        self.money_spent = 0.0

    def change_pnl(self, prev_price, cur_price, balance):
        self.pnl += (cur_price - prev_price) * balance

    # def update(self, asset_purchased, money_spent):
    #     self.asset_purchased += asset_purchased
    #     self.money_spent += money_spent

    # def unrealized_pnl(self, cur_amount, ask_price):
    #     if self.asset_purchased > 0:
    #         self.pnl += cur_amount * (ask_price - self.money_spent / self.asset_purchased)

    # def realized_pnl(self, amount_sold, price):
    #     if self.asset_purchased > 0:
    #         self.pnl += price - amount_sold * (self.money_spent / self.asset_purchased)

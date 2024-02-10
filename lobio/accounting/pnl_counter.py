class PnL_Counter:
    """Class for pnl counting."""

    def __init__(self):
        """Creation of pnl counter.
        
        PnL variable trackes PnL, asset_purchased equal to quote amount of trader and money_spent
        equal to traders money spent for quote.
        """
        self.pnl = 0.0
        # self.asset_purchased = 0.0
        # self.money_spent = 0.0

    def change_pnl(self, prev_price: float, cur_price: float, balance: float):
        """Calculation of PnL change due to exchange of trader.

        Args:
        ----
            prev_price (float): price of quote before exchange
            cur_price (float): price of quote after exchange
            balance (float): quote amount of trader
        """
        self.pnl += (cur_price - prev_price) * balance
    
    def reset(self):
        """Reseting pnl, setting to 0."""
        self.pnl = 0

    # def update(self, asset_purchased, money_spent):
    #     self.asset_purchased += asset_purchased
    #     self.money_spent += money_spent

    # def unrealized_pnl(self, cur_amount, ask_price):
    #     if self.asset_purchased > 0:
    #         self.pnl += cur_amount * (ask_price - self.money_spent / self.asset_purchased)

    # def realized_pnl(self, amount_sold, price):
    #     if self.asset_purchased > 0:
    #         self.pnl += price - amount_sold * (self.money_spent / self.asset_purchased)

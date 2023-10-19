PRICE_TICK: int = 2
AMOUNT_TICK: int = 8


class LimitOrder:
    def __init__(self, price: float, amount: float, side: int, trader_id: int):
        price = round(price, PRICE_TICK)
        amount = round(amount, AMOUNT_TICK)
        assert amount > 0
        assert price >= 0

        self.price = price
        self.amount = amount
        self.side = side
        self.trader_id = trader_id

    def __repr__(self):
        return f"LimitOrder({self.price}, {self.amount}, {self.side}, {self.trader_id})"

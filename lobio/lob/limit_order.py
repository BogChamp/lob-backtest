PRICE_TICK: int = 2
AMOUNT_TICK: int = 8


class LimitOrder:
    """Class for limit order imitation."""

    def __init__(self, base: float, quote: float, side: int, trader_id: int):
        """Creation of limit order.

        Args:
        ----
            base (float): amount of based asset desired per quoted asset
            quote (float): amount of quoted asset wanted to trade
            side (int): 1 - buy order, 0 - sell order
            trader_id (int): who set this limit order
        """
        base = round(base, PRICE_TICK)
        quote = round(quote, AMOUNT_TICK)
        assert quote > 0
        assert base >= 0

        self.base = base
        self.quote = quote
        self.side = side
        self.trader_id = trader_id

    def __repr__(self):
        return f"LimitOrder({self.base}, {self.quote}, {self.side}, {self.trader_id})"

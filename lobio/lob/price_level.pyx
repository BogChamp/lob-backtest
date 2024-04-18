class PriceLevelSimple:
    def __init__(self, base:int, quote: int) -> None:
        self.amount = [quote, 0]
        self.base = base
        self.my_order_id = None
    
    def __repr__(self) -> str:
        return f'PriceLevel{self.base, self.amount}'

    def add_liquidity(self, quote: int):
        if self.amount[1] != 0: # if we are in price level
            self.amount[1] += quote
        else:
            self.amount[0] += quote
    
    def total_amount(self) -> int:
        return self.amount[0] + self.amount[1]
    
    def execute_market_order(self, quote: int) -> Tuple[int, int|None]: # always keep in ind 1 amount with us. After we executed, move liquidity to ind 0.
        if quote <= self.amount[0]:
            self.amount[0] -= quote
            quote = 0
            me_executed = None
        else:
            quote -= self.amount[0]
            if quote <= self.amount[1]:
                self.amount[1] -= quote
                quote = 0
                self.amount = [self.amount[1], 0]
            else:
                quote -= self.amount[1]
                self.amount = [0, 0]
            me_executed = self.my_order_id
            self.my_order_id = None

        return quote, me_executed

    def change_historical_liquidity_opt(self, quote: int):
        if quote > 0:
            self.add_liquidity(quote)
        elif quote < 0: # from start to end
            quote = abs(quote)
            if quote <= self.amount[0]:
                self.amount[0] -= quote
            else:
                quote -= self.amount[0]
                self.amount[0] = 0
                self.amount[1] -= quote
    
    def change_historical_liquidity(self, quote: int):
        if quote > 0:
            self.add_liquidity(quote)
        elif quote < 0: # from start to end
            quote = abs(quote)
            if quote <= self.amount[1] - 1:
                self.amount[1] -= quote
            else:
                quote -= self.amount[1] - 1
                self.amount[1] = 1
                self.amount[0] -= quote

    def place_my_order(self, ratio_after_me: float, order_id: int):
        quote_after_me = int(ratio_after_me * self.total_amount())
        self.amount[0] = self.total_amount() - quote_after_me - 1
        self.amount[1] = quote_after_me + 1
        self.my_order_id = order_id

    def queue_dynamic(self, ratio_after_me: float):
        quote_after_me = int(ratio_after_me * self.amount[0])
        self.amount[0] -= quote_after_me
        self.amount[1] += quote_after_me
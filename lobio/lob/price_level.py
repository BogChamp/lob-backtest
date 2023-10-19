from .limit_order import LimitOrder, AMOUNT_TICK
from collections import defaultdict


class PriceLevelBase:
    def __init__(self, first_limit_order: LimitOrder):
        self.price = first_limit_order.price
        self.amount = first_limit_order.amount
        self.side = first_limit_order.side
        self.traders_order = [first_limit_order]

    def __repr__(self):
        return f"PriceLevel({self.price}, {self.amount}, {self.side})"

    def add_limit_order(self, limit_order: LimitOrder):  # trader_id: 0 - market, 1 - MM
        assert limit_order.price == self.price
        assert limit_order.side == self.side

        self.amount += limit_order.amount
        self.amount = round(self.amount, AMOUNT_TICK)
        if len(self.traders_order):
            if limit_order.trader_id == self.traders_order[-1].trader_id:
                self.traders_order[-1].amount += limit_order.amount
                self.traders_order[-1].amount = round(
                    self.traders_order[-1].amount, AMOUNT_TICK
                )
            else:
                self.traders_order.append(limit_order)
        else:
            self.traders_order.append(limit_order)

    def execute_limit_order(self, amount: float):
        remain_amount = round(amount, AMOUNT_TICK)
        match_info = defaultdict(int)  # trader_id - amount_sold

        for i, limit_order in enumerate(self.traders_order):
            match_info[limit_order.trader_id] += min(limit_order.amount, remain_amount)
            self.amount -= min(limit_order.amount, remain_amount)

            if remain_amount < limit_order.amount:
                limit_order.amount -= remain_amount
                limit_order.amount = round(limit_order.amount, AMOUNT_TICK)
                self.traders_order = self.traders_order[i:]
                remain_amount = 0
                break
            else:
                remain_amount -= limit_order.amount

        remain_amount = round(remain_amount, AMOUNT_TICK)
        self.amount = round(self.amount, AMOUNT_TICK)
        if self.amount == 0:
            self.traders_order = []

        return remain_amount, match_info


class PriceLevel(PriceLevelBase):
    def change_liquidity(self, amount: float, trader_id: int):
        if amount > 0:
            limit_order = LimitOrder(self.price, amount, self.side, trader_id)
            self.add_limit_order(limit_order)
        else:
            amount = abs(amount)
            for i, limit_order in enumerate(self.traders_order):
                if limit_order.side != trader_id:
                    continue
                self.amount -= min(limit_order.amount, amount)

                if amount < limit_order.amount:
                    limit_order.amount -= amount
                    limit_order.amount = round(limit_order.amount, AMOUNT_TICK)
                    break
                else:
                    amount -= limit_order.amount
                    del self.traders_order[i]

        self.amount = round(self.amount, AMOUNT_TICK)
        if self.amount == 0:
            self.traders_order = []

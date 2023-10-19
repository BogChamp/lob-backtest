import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from LimitOrder import LimitOrder
from OrderBook import OrderBookPrep, BUY, SELL, MARKET_ID


def read_raw_data():
    with open("diffs.json", "r", encoding="utf-8") as file_out:
        diffs = json.load(file_out)

    with open("trades.json", "r", encoding="utf-8") as file_out:
        trades = json.load(file_out)

    with open("init_lob.json", "r", encoding="utf-8") as file_out:
        init_lob = json.load(file_out)

    init_lob["bids"] = np.array(init_lob["bids"]).astype(float)
    init_lob["asks"] = np.array(init_lob["asks"]).astype(float)

    i = 0
    while diffs[i]["u"] <= init_lob["lastUpdateId"]:
        i += 1
    diffs = diffs[i:]

    i = 0
    while trades[i]["E"] <= diffs[0]["E"]:
        i += 1
    trades = trades[i:]

    return diffs, trades, init_lob


def separate_trades_by_diff(diffs, trades):
    trades_by_diff = []

    trades_index = 0
    for v in diffs[1:]:
        time_to = v["E"]
        trades_after_diff = []
        cur_trade = trades[trades_index]
        while cur_trade["T"] <= time_to:
            trades_after_diff.append(
                (cur_trade["T"], float(cur_trade["p"]), float(cur_trade["q"]))
            )
            trades_index += 1
            cur_trade = trades[trades_index]
        trades_by_diff.append(trades_after_diff)

    return trades_by_diff


def change_diffs(diffs):
    new_diffs = []
    for diff in diffs:
        new_diffs.append(
            (
                diff["E"],
                np.array(diff["b"]).astype(float),
                np.array(diff["a"]).astype(float),
            )
        )

    return new_diffs


def save_order_book(order_book):
    bids_prepared = []
    asks_prepared = []

    for bid in order_book.bids:
        bids_prepared.append([bid.price, bid.amount])

    for ask in order_book.asks:
        asks_prepared.append([ask.price, ask.amount])

    init_lob_prepared = {
        "lastUpdateId": new_diffs[0][0],
        "bids": bids_prepared,
        "asks": asks_prepared,
    }

    with open("init_lob_prepared.json", "w") as fp:
        json.dump(init_lob_prepared, fp)


def prepare_trades_diffs(order_book, new_diffs, trades_by_diff):
    trades_prepared = []
    diffs_prepared = []

    for i, diff in enumerate(tqdm(new_diffs[1:])):
        cur_trades = trades_by_diff[i]
        for trade in cur_trades:
            if trade[1] >= order_book.ask_price():
                side = BUY
            elif trade[1] <= order_book.bid_price():
                side = SELL
            order_book.set_order(LimitOrder(trade[1], trade[2], side, MARKET_ID))
            trades_prepared.append([trade[0], trade[1], trade[2], side])
        diffs_prepared.append(order_book.track_diff(diff))

    return trades_prepared, diffs_prepared


if __name__ == "__main__":
    diffs, trades, init_lob = read_raw_data()
    trades_by_diff = separate_trades_by_diff(diffs, trades)
    new_diffs = change_diffs(diffs)

    order_book = OrderBookPrep.create_lob_init(init_lob)
    order_book.track_diff(new_diffs[0])
    save_order_book(order_book)

    trades_prepared, diffs_prepared = prepare_trades_diffs(
        order_book, new_diffs, trades_by_diff
    )

    trades_prepared = pd.DataFrame(
        trades_prepared, columns=["timestamp", "price", "amount", "side"]
    )
    trades_prepared.to_csv("trades_prepared.csv", index=False)

    with open("diffs_prepared.json", "w") as fp:
        json.dump(diffs_prepared, fp)

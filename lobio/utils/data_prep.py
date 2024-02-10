import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from lobio.lob.limit_order import LimitOrder
from lobio.lob.order_book import OrderBookPrep, Side, TraderId

from typing import Tuple, Sequence

def read_raw_data() -> Tuple[Sequence[dict], Sequence[dict], dict]:
    """Read raw data, fetched from exchange.

    Returns
    -------
        Tuple[Sequence[dict], Sequence[dict], dict]: list of incremental diffs, list of trades and lob state.
        Incremental diffs and trades happened after lob state.
    """
    with open("./data/diffs.json", "r", encoding="utf-8") as file_out:
        diffs = json.load(file_out)

    with open("./data/trades.json", "r", encoding="utf-8") as file_out:
        trades = json.load(file_out)

    with open("./data/init_lob.json", "r", encoding="utf-8") as file_out:
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


def separate_trades_by_diff(diffs: Sequence[dict], 
                            trades: Sequence[dict]) -> Sequence[Sequence[Tuple[float, float, float, int]]]:
    """Distribute all trades to incremental diffs.

    Args:
    ----
        diffs (Sequence[dict]): list of diffs.
        trades (Sequence[dict]): list of trades.

    Returns:
    -------
        Sequence[Sequence]: list with length equal to diffs, each element of it is list with trades,
        happened before current diff and after previous. Trade info containes timestamp in nanosecond, base and quote.
    """
    trades_by_diff = []

    trades_index = 0
    trades.append(
        {"T": np.inf}
    )  # plug for case if all trades performed before last incremental diff
    for v in diffs[1:]:
        time_to = v["E"]
        trades_after_diff = []
        cur_trade = trades[trades_index]
        while cur_trade["T"] <= time_to:
            trades_after_diff.append(
                (cur_trade["T"], float(cur_trade["p"]), float(cur_trade["q"]), int(cur_trade["m"]))
            )
            trades_index += 1
            cur_trade = trades[trades_index]
        trades_by_diff.append(trades_after_diff)

    return trades_by_diff


def change_diffs(diffs: Sequence[dict]) -> Sequence[Tuple[float, Sequence[Sequence[float]], Sequence[Sequence[float]]]]:
    """Remove not necessary data from raw diff data from exchange.

    Args:
    ----
        diffs (Sequence[dict]): list of diffs.

    Returns:
    -------
        Sequence[Tuple[float, Sequence, Sequence]]: list, each element of it containes
        timestamp of diff in nanoseconds, list of bid diffs and ask diffs.
    """
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


def save_order_book(order_book: OrderBookPrep):
    """Save lob state with applied first loaded diff.

    Saves last lob update timestamp in nanoseconds and list of bids and asks. 

    Args:
    ----
        order_book (OrderBookPrep): lob state.
    """
    bids_prepared = []
    asks_prepared = []

    for bid in order_book.bids:
        bids_prepared.append([bid.base, bid.quote])

    for ask in order_book.asks:
        asks_prepared.append([ask.base, ask.quote])

    init_lob_prepared = {
        "lastUpdateId": new_diffs[0][0],
        "bids": bids_prepared,
        "asks": asks_prepared,
    }

    with open("./data/init_lob_prepared.json", "w") as fp:
        json.dump(init_lob_prepared, fp)


def prepare_trades_diffs(order_book: OrderBookPrep, 
                         new_diffs: Sequence[Tuple[float, Sequence[Sequence[float]], Sequence[Sequence[float]]]], 
                         trades_by_diff: Sequence[Sequence[Tuple[float, float, float, int]]]) \
                            -> Tuple[Sequence[Sequence[float]], 
                                     Sequence[Tuple[float, Sequence[Sequence[float]], 
                                                    Sequence[Sequence[float]]]]]:
    """Prepare data for trades and diffs in convenient format.

    For trades leave only timestamp in nanoseconds, price, amount and side.
    For diffs leave timestamp in nanoseconds, list of bids and asks updates, containing price and value change.

    Returns
    -------
        tuple: trades data and diffs data without trades influence.
    """
    trades_prepared = []
    diffs_prepared = []

    for i, diff in enumerate(tqdm(new_diffs[1:])):
        cur_trades = trades_by_diff[i]
        for trade in cur_trades:
            side = trade[3]
            order_book.set_limit_order(
                LimitOrder(trade[1], trade[2], side, TraderId.MARKET)
            )
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
    trades_prepared.to_csv("./data/trades_prepared.csv", index=False)

    with open("./data/diffs_prepared.json", "w") as fp:
        json.dump(diffs_prepared, fp)

import json
import numpy as np
import polars as pl
from typing import Sequence, Tuple
from lobio.lob.limit_order import PRICE_TICK, AMOUNT_TICK
from lobio.lob.price_level import Side
from lobio.lob.order_book import OrderBookPrep, OrderBook

def read_raw_data() -> Tuple[Sequence[dict], Sequence[dict], dict]:
    """Read raw data, fetched from exchange.

    Returns
    -------
        Tuple[Sequence[dict], Sequence[dict], dict]: list of incremental diffs, list of trades and lob state.
        Incremental diffs and trades happened after lob state.
    """
    with open("./data/diffs.json", "r", encoding="utf-8") as file_out:
        diffs = json.load(file_out)

    with open("./data/aggtrades.json", "r", encoding="utf-8") as file_out:
        aggtrades = json.load(file_out)

    with open("./data/init_lob.json", "r", encoding="utf-8") as file_out:
        init_lob = json.load(file_out)

    init_lob["bids"] = np.array(init_lob["bids"]).astype(float)
    init_lob["asks"] = np.array(init_lob["asks"]).astype(float)

    return diffs, aggtrades, init_lob

def check_aggtrades(aggtrades: Sequence[dict]) -> None:
    prev_T = 0
    prev_l = aggtrades[0]['f'] - 1
    for aggtrade in aggtrades:
        assert aggtrade['T'] >= prev_T
        assert aggtrade['f'] == prev_l + 1
        prev_T = aggtrade['T']
        prev_l = aggtrade['l']

def check_diffs(diffs: Sequence[dict]) -> None:
    prev_T = diffs[0]['T']
    prev_u = diffs[0]['u']
    for j, diff in enumerate(diffs[1:]):
        assert diff['T'] >= prev_T
        assert diff['U'] >= prev_u + 1, j
        assert diff['pu'] == prev_u
        prev_u = diff['u']
        prev_T = diff['T']

def cut_diffs(diffs: Sequence[dict], UpdateId: int) -> Sequence[dict]:
    i = 0
    while diffs[i]["u"] < UpdateId:
        i += 1

    if len(diffs) > i:
        if diffs[i]["U"] > UpdateId or diffs[i]["u"] < UpdateId:
            raise Exception("CORRUPTED DATA. DIFF. DEPTH are not consistent.")
    else:
        raise Exception("CORRUPTED DATA. DIFF. DEPTH are not consistent.")
    diffs = diffs[i:]

    return diffs

def cut_trades(trades: Sequence[dict], timestamp: int) -> Sequence[dict]:
    i = 0
    while trades[i]['T'] <= timestamp:
        i += 1
    trades = trades[i:]
    return trades

def convert_diffs(diffs: Sequence[dict]) -> Sequence[Tuple[int, Sequence[Tuple[float, float]], Sequence[Tuple[float, float]]]]:
    new_diffs = []
    for diff in diffs:
        cur_bids = np.array(diff["b"]).astype(float)
        cur_asks = np.array(diff["a"]).astype(float)
        new_diffs.append(
            (
                diff["T"],
                cur_bids,
                cur_asks,
            )
        )
    return new_diffs

def compress_diffs(diffs: Sequence[Tuple[int, Sequence[Tuple[float, float]], Sequence[Tuple[float, float]]]]) -> pl.DataFrame:
    diff_sequence = []
    for diff in diffs:
        ts = diff[0]
        bids = diff[1]
        asks = diff[2]

        for bid in bids[::-1]:
            diff_sequence.append((ts, bid[0], bid[1], Side.BUY))

        for ask in asks:
            diff_sequence.append((ts, ask[0], ask[1], Side.SELL))

    diffs_prepared = pl.DataFrame(diff_sequence, ('T', 'base', 'quote', 'side'))
    diffs_prepared = diffs_prepared.with_columns((pl.col('base') * 10**PRICE_TICK).round().cast(pl.UInt32), 
                       (pl.col('quote') * 10**AMOUNT_TICK).round().cast(pl.UInt64), 
                       pl.col('side').cast(pl.Int8),
                       pl.col('T').cast(pl.UInt64))
    return diffs_prepared

def compress_init_lob(order_book: OrderBook, timestamp: int) -> np.ndarray[int]:
    bids_prepared = []
    asks_prepared = []
    min_len = min(len(order_book.bids), len(order_book.asks))

    for bid in order_book.bids[:min_len]:
        bids_prepared.append([bid.base, bid.quote])

    for ask in order_book.asks[:min_len]:
        asks_prepared.append([ask.base, ask.quote])

    init_lob_data = np.concatenate([bids_prepared, asks_prepared], axis=1)
    init_lob_data[:, [0, 2]] *= 10**PRICE_TICK
    init_lob_data[:, [1, 3]] *= 10**AMOUNT_TICK
    init_lob_data = np.round(init_lob_data)
    init_lob_data = np.append(init_lob_data, [[timestamp] * init_lob_data.shape[1]], axis=0).astype(int)

    return init_lob_data

def compress_trades(trades: Sequence[dict]) -> pl.DataFrame:
    trades_raw = []
    for trade in trades:
        side = Side.SELL if trade['m'] else Side.BUY
        trades_raw.append((trade['T'], float(trade['p']), float(trade['q']), side))

    trades_raw = pl.DataFrame(trades_raw, ('T', 'base', 'quote', 'side'))
    trades_raw = trades_raw.with_columns((pl.col('base') * 10**PRICE_TICK).round().cast(pl.UInt32), 
                        (pl.col('quote') * 10**AMOUNT_TICK).round().cast(pl.UInt64), 
                        pl.col('side').cast(pl.Int8),
                        pl.col('T').cast(pl.UInt64))
    return trades_raw

if __name__ == "__main__":
    diffs, aggtrades, init_lob = read_raw_data()

    check_aggtrades(aggtrades)
    check_diffs(diffs)
    diffs = cut_diffs(diffs, init_lob["lastUpdateId"])
    aggtrades = cut_trades(aggtrades, diffs[0]["T"])

    new_diffs = convert_diffs(diffs)
    order_book = OrderBookPrep.create_lob_init(init_lob)
    order_book.track_diff(new_diffs[0])
    new_diffs = new_diffs[1:]

    compressed_init_lob = compress_init_lob(order_book, diffs[0]['T'])
    compressed_diffs = compress_diffs(new_diffs)
    compressed_trades = compress_trades(aggtrades)

    with open('./data/init_lob_prepared.npy', 'wb') as f:
        np.save(f, compressed_init_lob)
    compressed_diffs.write_parquet("./data/diffs_prepared.parquet")
    compressed_trades.write_parquet("./data/aggtrades_raw.parquet")

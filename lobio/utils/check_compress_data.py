import json
import numpy as np
from typing import Sequence, Tuple
from loguru import logger
from tqdm import tqdm

from lobio.lob.limit_order import PRICE_TICK, AMOUNT_TICK, Side
from lobio.lob.order_book import OrderBookSimple

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
    for diff in tqdm(diffs):
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

def compress_diffs(diffs: Sequence[Tuple[int, Sequence[Tuple[float, float]], Sequence[Tuple[float, float]]]]) -> np.ndarray[int]:
    diff_sequence = []
    for diff in tqdm(diffs):
        ts = diff[0]
        bids = diff[1]
        asks = diff[2]

        for bid in bids[::-1]:
            diff_sequence.append((ts, bid[0], bid[1], Side.BUY))

        for ask in asks:
            diff_sequence.append((ts, ask[0], ask[1], Side.SELL))

    diffs_prepared = np.array(diff_sequence)
    diffs_prepared[:, 1] *= 10**PRICE_TICK
    diffs_prepared[:, 2] *= 10**AMOUNT_TICK
    return diffs_prepared.round().astype(int)

def compress_init_lob(order_book: OrderBookSimple, timestamp: int) -> np.ndarray[int]:
    min_len = min(len(order_book.bids), len(order_book.asks))
    bids_prepared = order_book.bids[:min_len]
    asks_prepared = order_book.asks[:min_len]

    init_lob_data = np.concatenate([bids_prepared, asks_prepared], axis=1)
    init_lob_data[:, [0, 2]] *= 10**PRICE_TICK
    init_lob_data[:, [1, 3]] *= 10**AMOUNT_TICK
    init_lob_data = np.round(init_lob_data)
    init_lob_data = np.append(init_lob_data, [[timestamp] * init_lob_data.shape[1]], axis=0).astype(int)

    return init_lob_data

def compress_trades(trades: Sequence[dict]) -> np.ndarray[int]:
    trades_raw = []
    for trade in tqdm(trades):
        side = Side.SELL if trade['m'] else Side.BUY
        trades_raw.append((trade['T'], float(trade['p']), float(trade['q']), side))

    trades_raw = np.array(trades_raw)
    trades_raw[:, 1] *= 10**PRICE_TICK
    trades_raw[:, 2] *= 10**AMOUNT_TICK
    return trades_raw.round().astype(int)

if __name__ == "__main__":
    diffs, aggtrades, init_lob = read_raw_data()

    check_aggtrades(aggtrades)
    check_diffs(diffs)
    diffs = cut_diffs(diffs, init_lob["lastUpdateId"])
    aggtrades = cut_trades(aggtrades, diffs[0]["T"])

    logger.info("converting diffs to easier format")
    new_diffs = convert_diffs(diffs)
    order_book = OrderBookSimple.create_lob_init(init_lob)
    order_book.apply_historical_update(new_diffs[0])
    new_diffs = new_diffs[1:]

    logger.info("compressing initial lob state")
    compressed_init_lob = compress_init_lob(order_book, diffs[0]['T'])
    logger.info("compressing diffs")
    compressed_diffs = compress_diffs(new_diffs)
    logger.info("compressing trades")
    compressed_trades = compress_trades(aggtrades)

    with open('./data/init_lob_raw.npy', 'wb') as f:
        np.save(f, compressed_init_lob)
    with open('./data/diffs_raw.npy', 'wb') as f:
        np.save(f, compressed_diffs)
    with open('./data/aggtrades_raw.npy', 'wb') as f:
        np.save(f, compressed_trades)
    logger.info("check and compression completed!")

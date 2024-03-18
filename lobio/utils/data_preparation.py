import polars as pl
import numpy as np

from lobio.utils.utils import group_diffs, get_initial_order_book, merge_orders, \
    check_if_sorted, group_historical_trades, find_unseen_dynamic_of_lob, group_orders

if __name__ == "__main__":
    diffs_prepared_file = "./data/diffs_prepared.parquet"
    init_lob_prepared_file = "./data/init_lob_prepared.npy"
    aggtrades_file = "./data/aggtrades_raw.parquet"

    diffs = pl.read_parquet(diffs_prepared_file)
    with open(init_lob_prepared_file, "rb") as file:
        init_lob = np.load(file)
    aggtrades_raw = pl.read_parquet(aggtrades_file)

    diffs_grouped = group_diffs(diffs)
    check_if_sorted(diffs_grouped)

    aggtrades_raw = aggtrades_raw.to_numpy().astype(int)
    trades_per_diff = group_historical_trades(aggtrades_raw, diffs_grouped)
    trades_prepared, additional_data = find_unseen_dynamic_of_lob(init_lob, trades_per_diff, diffs_grouped)
    orders_seq = merge_orders(trades_prepared, additional_data)
    orders_prepared = pl.DataFrame(orders_seq, schema=('T', 'diff_num', 'base', 'quote', 'side'))
    orders_prepared.write_parquet("./data/orders_prepared.parquet")

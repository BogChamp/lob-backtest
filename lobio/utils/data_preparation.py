import numpy as np
from loguru import logger

from lobio.utils.utils import group_diffs, merge_orders, check_if_diffs_sorted, \
    group_historical_trades, find_unseen_dynamic_of_lob, prepare_diffs, update_diffs, \
    split_unseen_dynamic, prepare_init_lob, aggregate_orders, cut_diffs, group_orders

if __name__ == "__main__":
    diffs_raw_file = "./data/diffs_raw.npy"
    init_lob_raw_file = "./data/init_lob_raw.npy"
    aggtrades_raw_file = "./data/aggtrades_raw.npy"

    with open(init_lob_raw_file, 'rb') as file:
        init_lob = np.load(file)
    with open(diffs_raw_file, 'rb') as file:
        diffs = np.load(file)
    with open(aggtrades_raw_file, 'rb') as file:
        aggtrades = np.load(file)

    logger.info("Grouping diffs")
    diffs_grouped = group_diffs(diffs)
    check_if_diffs_sorted(diffs_grouped)

    logger.info("Grouping aggtrades")
    trades_per_diff = group_historical_trades(aggtrades, diffs_grouped)
    
    logger.info("Finding unseen dynamic")
    trades_info, unseen_dynamic = find_unseen_dynamic_of_lob(init_lob, trades_per_diff, diffs_grouped)

    logger.info("Splitting unseen dynamic events")
    unseen_diffs, orders, initial_state_update = split_unseen_dynamic(unseen_dynamic)

    logger.info("Preparing initial lob state")
    init_lob_prepared = prepare_init_lob(init_lob, initial_state_update)

    logger.info("Updating diffs")
    new_diffs = update_diffs(unseen_diffs, init_lob_prepared, diffs_grouped)

    logger.info("Merging orders")
    trades_info.sort(key=lambda x: (x[0], x[1]))
    orders.sort(key=lambda x: (x[0], x[1]))
    orders_flow = merge_orders(trades_info, orders)

    logger.info("Squeezing orders")
    orders_prepared = aggregate_orders(orders_flow)

    logger.info("Grouping orders between diffs")
    orders_per_diff = group_orders(orders_prepared, len(new_diffs))

    logger.info("Cutting diffs")
    diffs_cut = cut_diffs(init_lob_prepared, new_diffs, orders_per_diff)

    logger.info("Preparing diffs")
    diffs_prepared = prepare_diffs(diffs_cut)

    with open('./data/init_lob_prepared.npy', 'wb') as f:
        np.save(f, init_lob_prepared)
    with open('./data/diffs_prepared.npy', 'wb') as f:
        np.save(f, diffs_prepared)
    with open('./data/orders_prepared.npy', 'wb') as f:
        np.save(f, orders_prepared)

    logger.info("Data preparation is done!")

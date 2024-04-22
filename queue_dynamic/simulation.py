import numpy as np
from typing import Tuple, Sequence
from bisect import bisect_left
import torch.nn as nn
import torch

from lobio.lob.limit_order import Side, OrderType, Order
from lobio.lob.price_level import PriceLevelSimple
from lobio.utils.utils import get_initial_order_book_simple2
from queue_dynamic.models.reinforce_model import GaussianPDFModel

def run_true_simulation(init_lob: np.ndarray[int], 
                        diffs_grouped: Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]], 
                        orders_per_diff: Sequence[Sequence[Tuple[int, Order]]],
                        samples: np.array,
                        place_ratios: np.array,
                        rng: np.random.Generator) -> list[int|None|Tuple[int, int]]:
    poses_true_info = [None] * len(samples)
    my_current_poses: dict[int, PriceLevelSimple] = {}
    sample_running_ind = 0
    ob = get_initial_order_book_simple2(init_lob)

    for i, diff in enumerate(diffs_grouped):
        while sample_running_ind < len(samples) and samples[sample_running_ind][0] == i:
            if samples[sample_running_ind][2] == Side.BUY:
                ind = bisect_left(ob.bids, -samples[sample_running_ind][1], key=lambda x: -x.base)
                if ob.bids[ind].my_order_id != None:
                    my_current_poses.pop(ob.bids[ind].my_order_id)
                    poses_true_info[ob.bids[ind].my_order_id] = -1 # THINK

                ob.bids[ind].place_my_order(place_ratios[sample_running_ind], sample_running_ind)
                my_current_poses[sample_running_ind] = ob.bids[ind]
            else:
                ind = bisect_left(ob.asks, samples[sample_running_ind][1], key=lambda x: x.base)
                if ob.asks[ind].my_order_id != None:
                    my_current_poses.pop(ob.asks[ind].my_order_id)
                    poses_true_info[ob.asks[ind].my_order_id] = -1 # THINK

                ob.asks[ind].place_my_order(place_ratios[sample_running_ind], sample_running_ind)
                my_current_poses[sample_running_ind] = ob.asks[ind]

            sample_running_ind += 1

        cur_orders = orders_per_diff[i]
        for j, (_, order) in enumerate(cur_orders):
            if order.type == OrderType.MARKET:
                my_orders_eaten = ob.set_market_order([order.quote, order.side])
                for my_eaten in my_orders_eaten:
                    poses_true_info[my_eaten] = (i, j)
                    my_current_poses.pop(my_eaten)
            else:
                ob.set_limit_order([order.base, order.quote, order.side])

        ratios = np.zeros(len(my_current_poses))#ratios = rng.beta(1.5, 12, size=len(my_current_poses)) # HERE UNCERTAINTY
        for j, p_l in enumerate(my_current_poses.values()):
            p_l.queue_dynamic(ratios[j])

        my_orders_removed = ob.apply_historical_update(diff)
        for my_removed in my_orders_removed:
            my_current_poses.pop(my_removed)
    
    return poses_true_info

def run_pred_simulation(init_lob: np.ndarray[int],
                        diffs_grouped: Sequence[Tuple[int, Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]],
                        orders_per_diff: Sequence[Sequence[Tuple[int, Order]]],
                        samples: np.array, 
                        place_ratios: np.array, 
                        rl_model: GaussianPDFModel) -> Tuple[list[int|None|Tuple[int, int]], list[list[Tuple[int, int, int]]]]:
    poses_pred_info = [None] * len(samples)
    my_current_poses_info: dict[int, Tuple[PriceLevelSimple, int, int, int]] = {} # order_id - price level, number of diffs lived,amount changed, old_amount
    obs_actions = [[] for _ in range(len(samples))]
    sample_running_ind = 0
    ob = get_initial_order_book_simple2(init_lob)

    for i, diff in enumerate(diffs_grouped):
        while sample_running_ind < len(samples) and samples[sample_running_ind][0] == i:
            if samples[sample_running_ind][2] == Side.BUY:
                ind = bisect_left(ob.bids, -samples[sample_running_ind][1], key=lambda x: -x.base) # INDEXES FOUND IN PREV STEP(WHILE GROUND TRUE GENERATED)
                if ob.bids[ind].my_order_id != None:
                    my_current_poses_info.pop(ob.bids[ind].my_order_id)
                    poses_pred_info[ob.bids[ind].my_order_id] = -1 # THINK
                    obs_actions[ob.bids[ind].my_order_id] = []

                ob.bids[ind].place_my_order(place_ratios[sample_running_ind], sample_running_ind)
                my_current_poses_info[sample_running_ind] = [ob.bids[ind], 0, 0, ob.bids[ind].total_amount()]
            else:
                ind = bisect_left(ob.asks, samples[sample_running_ind][1], key=lambda x: x.base)
                if ob.asks[ind].my_order_id != None:
                    my_current_poses_info.pop(ob.asks[ind].my_order_id)
                    poses_pred_info[ob.asks[ind].my_order_id] = -1 # THINK
                    obs_actions[ob.asks[ind].my_order_id] = []

                ob.asks[ind].place_my_order(place_ratios[sample_running_ind], sample_running_ind)
                my_current_poses_info[sample_running_ind] = [ob.asks[ind], 0, 0, ob.asks[ind].total_amount()]

            sample_running_ind += 1

        cur_orders = orders_per_diff[i]
        for j, (_, order) in enumerate(cur_orders):
            if order.type == OrderType.MARKET:
                my_orders_eaten = ob.set_market_order([order.quote, order.side])
                for my_eaten in my_orders_eaten:
                    poses_pred_info[my_eaten] = (i, j)
                    my_current_poses_info.pop(my_eaten)
            else:
                ob.set_limit_order([order.base, order.quote, order.side])

        observations = []
        for info in my_current_poses_info.values():
            observations.append((info[1], info[2]))

        if len(observations):
            observations_tensor = torch.FloatTensor(observations)
            ratios = rl_model.sample(observations_tensor)
            for j, (order_id, info) in enumerate(my_current_poses_info.items()):
                info[0].queue_dynamic(ratios[j])
                obs_actions[order_id].append((info[1], info[2], ratios[j]))

        my_orders_removed = ob.apply_historical_update(diff)
        for my_removed in my_orders_removed:
            my_current_poses_info.pop(my_removed)

        for info in my_current_poses_info.values():
            info[1] += 1
            info[2] = info[0].total_amount() - info[3]
            info[3] = info[0].total_amount()
    
    return poses_pred_info, obs_actions

from typing import Tuple
import torch

from queue_dynamic.models.reinforce_model import GaussianPDFModel

def calculate_loss(poses_true_info: list[None|Tuple[int, int]], 
                   poses_pred_info: list[None|Tuple[int, int]], 
                   obs_actions: list[list[Tuple[int, int, int]]], 
                   rl_model: GaussianPDFModel,
                   gamma: float) -> torch.FloatTensor:
    max_len = 0
    all_episod_records = []
    for episod_record in obs_actions:
        if len(episod_record) > max_len:
            max_len = len(episod_record)
        
        all_episod_records.append(torch.tensor(episod_record))
    
    all_episod_records = torch.concat(all_episod_records, dim=0)
    all_log_probs = rl_model.log_probs(all_episod_records)

    loss = 0
    M = 0
    running_idx = 0
    for i, episod_record in enumerate(obs_actions):
        if not (poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == -1):
            if len(episod_record):
                coefs = gamma**torch.arange(0, len(episod_record))
                cum_sum = torch.cumsum(coefs, dim=0)
                reverse_cum_sum = coefs - cum_sum + cum_sum[-1]
                episod_reward = 2 * (poses_true_info[i] != poses_pred_info[i]) - 1 # -1 for true
                loss += episod_reward * (reverse_cum_sum * all_log_probs[running_idx:running_idx+len(episod_record)]).sum() / len(episod_record)
                M += 1
        running_idx += len(episod_record)

    loss /= M

    return loss

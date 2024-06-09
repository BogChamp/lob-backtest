from typing import Tuple
import torch

from queue_dynamic.models.models import GaussianPDFModel, ModelPerceptron

def reinforce_objective(poses_true_info: list[int|None|Tuple[int, int]], 
                        poses_pred_info: list[int|None|Tuple[int, int]], 
                        obs_actions: list[list[Tuple[int, int, int]]], 
                        rl_model: GaussianPDFModel,
                        gamma: float) -> Tuple[torch.FloatTensor, float]:

    all_episod_records = []
    for episod_record in obs_actions: # group for applying log prob once
        all_episod_records.append(torch.FloatTensor(episod_record))

    all_episod_records = torch.concat(all_episod_records, dim=0)
    all_log_probs = rl_model.log_probs(all_episod_records)

    loss = 0
    reward = 0
    M = 0
    running_idx = 0 
    for i, episod_record in enumerate(obs_actions):
        # if during simulations orders were overplaced or not traded
        if not ((poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == -1) or \
                (poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == None)): 
            if len(episod_record):
                reward += poses_true_info[i] == poses_pred_info[i]

                coefs = gamma**torch.arange(0, len(episod_record))
                cum_sum = torch.cumsum(coefs, dim=0)
                reverse_cum_sum = coefs - cum_sum + cum_sum[-1]
                episod_cost = 2 * (poses_true_info[i] != poses_pred_info[i]) - 1 # -1 for true
                loss += episod_cost * (reverse_cum_sum * all_log_probs[running_idx:running_idx+len(episod_record)]).sum() / len(episod_record)
                M += 1
        running_idx += len(episod_record)

    loss /= M
    reward /= M

    return loss, reward

def reinforce_objective_with_baseline(poses_true_info: list[int|None|Tuple[int, int]], 
                   poses_pred_info: list[int|None|Tuple[int, int]], 
                   obs_actions: list[list[Tuple[int, int, int]]], 
                   rl_model: GaussianPDFModel,
                   gamma: float,
                   baseline: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, float, torch.FloatTensor]:

    all_episod_records = []
    max_episod_len = 0 # in current iter
    for episod_record in obs_actions:
        if len(episod_record) > max_episod_len:
            max_episod_len = len(episod_record)
        all_episod_records.append(torch.FloatTensor(episod_record))

    all_episod_records = torch.concat(all_episod_records, dim=0)
    all_log_probs = rl_model.log_probs(all_episod_records)

    loss = 0
    reward = 0
    M = 0
    running_idx = 0
    new_baseline = torch.zeros(max_episod_len)
    if baseline is None:
        baseline = torch.FloatTensor([])
    for i, episod_record in enumerate(obs_actions):
        # if during simulations orders were overplaced
        if not ((poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == -1) or \
                (poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == None)): 
            if len(episod_record):
                reward += poses_true_info[i] == poses_pred_info[i]

                coefs = gamma**torch.arange(0, len(episod_record))
                cum_sum = torch.cumsum(coefs, dim=0)
                reverse_cum_sum = coefs - cum_sum + cum_sum[-1]
                episod_cost = 2 * (poses_true_info[i] != poses_pred_info[i]) - 1 # -1 for true
                episod_costs = episod_cost * reverse_cum_sum / len(episod_record) # accumulated reward per step
                len_to_apply_baseline = min(len(episod_costs), len(baseline)) # max len of prev baseline can be less or more
                episod_costs[:len_to_apply_baseline] -= baseline[:len_to_apply_baseline]
                loss += (episod_costs * all_log_probs[running_idx:running_idx+len(episod_record)]).sum()
                M += 1

                new_baseline[:len(episod_record)] += episod_costs
        running_idx += len(episod_record)

    loss /= M
    reward /= M
    new_baseline /= M
    return loss, reward, new_baseline

def critic_objective_td(costs_td: torch.FloatTensor,
                     values: torch.FloatTensor,
                     gamma: float,
                     N_td : int):
    
    objective = (values[:-N_td] - costs_td - values[N_td:] * gamma**N_td)**2
    return objective.mean()

def critic_objective_mse(costs_mse: torch.FloatTensor,
                     values: torch.FloatTensor):
    
    objective = (values - costs_mse)**2
    return objective.mean()

def actor_objective(all_episod_records: torch.FloatTensor,
                    poses_true_info: list[int|None|Tuple[int, int]],
                    poses_pred_info: list[int|None|Tuple[int, int]],
                    obs_actions: list[list[Tuple[int, int, int]]],
                    all_costs_per_step: list[torch.FloatTensor],
                    rl_agent: GaussianPDFModel,
                    rl_critic: ModelPerceptron,
                    gamma: float
                    ) -> torch.float:
    all_log_probs = rl_agent.log_probs(all_episod_records)
    values = rl_critic(all_episod_records[:, :-1]).detach().reshape(-1, )
    running_idx = 0
    loss = 0
    M = 0
    for i, episod_record in enumerate(obs_actions):
        if not ((poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == -1) or \
                (poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == None)):
            if len(episod_record) > 1:
                episod_costs = all_costs_per_step[i]
                episod_values = values[running_idx:running_idx+len(episod_record)]
                episod_log_probs = all_log_probs[running_idx:running_idx+len(episod_record)]
                episod_A = (episod_costs - episod_values)[:-1] + gamma*episod_values[1:]
                loss += (gamma**torch.arange(0, len(episod_record)-1) * episod_A * episod_log_probs[:-1]).sum()
                M += 1

        running_idx += len(episod_record)

    loss /= M

    return loss

def ppo_objective(all_episod_records: torch.FloatTensor,
                    poses_true_info: list[int|None|Tuple[int, int]],
                    poses_pred_info: list[int|None|Tuple[int, int]],
                    obs_actions: list[list[Tuple[int, int, int]]],
                    all_costs_per_step: list[torch.FloatTensor],
                    rl_agent: GaussianPDFModel,
                    rl_critic: ModelPerceptron,
                    gamma: float,
                    epsilon: float = 0.2):
    old_log_probs = rl_agent.log_probs(all_episod_records).detach()
    all_log_probs = rl_agent.log_probs(all_episod_records)
    values = rl_critic(all_episod_records[:, :-1]).detach().reshape(-1, )
    running_idx = 0
    loss = 0
    M = 0
    for i, episod_record in enumerate(obs_actions):
        if not ((poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == -1) or \
                (poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == None)):
            if len(episod_record) > 1:
                episod_costs = all_costs_per_step[i]
                episod_values = values[running_idx:running_idx+len(episod_record)]
                episod_log_probs = all_log_probs[running_idx:running_idx+len(episod_record)]
                episod_old_log_probs = old_log_probs[running_idx:running_idx+len(episod_record)]
                prob_ratio = torch.exp(episod_log_probs - episod_old_log_probs)[:-1]
                prob_ratio_clip = torch.clip(prob_ratio, 1-epsilon, 1+epsilon)
                episod_A = (episod_costs - episod_values)[:-1] + gamma*episod_values[1:]
                loss += (gamma**torch.arange(0, len(episod_record)-1) * torch.max(episod_A * prob_ratio, episod_A * prob_ratio_clip)).sum()
                M += 1

        running_idx += len(episod_record)

    loss /= M
    return loss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import torch

from lobio.utils.utils import group_diffs, group_orders
from queue_dynamic.models.models import GaussianPDFModel, ModelPerceptron, Critic
from queue_dynamic.simulation import run_true_simulation, run_pred_simulation
from queue_dynamic.losses import critic_objective_td, critic_objective_mse, actor_objective, ppo_objective

sns.set_theme(style="darkgrid")


diffs_prepared_file = "../data/diffs_prepared.npy"
init_lob_prepared_file = "../data/init_lob_prepared.npy"
orders_prepared_file = "../data/orders_prepared.npy"

pl_to_enter_file = "../data/price_level_to_enter.npy"

with open(init_lob_prepared_file, 'rb') as file:
    init_lob = np.load(file)
with open(diffs_prepared_file, 'rb') as file:
    diffs = np.load(file)
with open(orders_prepared_file, 'rb') as file:
    orders = np.load(file)

with open(pl_to_enter_file, 'rb') as file:
    pl_to_enter = np.load(file)

diffs_grouped = group_diffs(diffs)
orders_per_diff = group_orders(orders, len(diffs_grouped))
n_poses = len(pl_to_enter)

dim_observation = 3
dim_action = 1
n_hidden_layers = 1
dim_hidden = 8
n_hidden_layers_critic = 1
dim_hidden_critic = 8
std = 0.01
scale_factor = 10
gamma = 0.95
lr = 0.001
lr_critic = 0.005
N_iter = 40
N_episode = 500
N_td = 1
N_critic_epoch = 30
#epsilon = 0.2
#N_ppo_epoch = 30
n_exp = 10

params = {'dim_obs': dim_observation, 
          'dim_act': dim_action, 
          'dim_hid': dim_hidden,
          'n_hid_layers': n_hidden_layers,
          'n_hidden_layers_critic': n_hidden_layers_critic,
          'dim_hidden_critic': dim_hidden_critic,
          'std': std, 
          'scale_factor': scale_factor,
          'gamma': gamma,
          'lr': lr,
          'lr_critic': lr_critic,
          'N_iter': N_iter,
          'N_episod': N_episode,
          'n_exp': n_exp,
          'N_td': N_td,
          'N_critic_epoch': N_critic_epoch,
          'dynamic': 'zero'
          }


seeds = [int(round(np.random.uniform(0, 10), 5) * 10**5) for _ in range(n_exp)]
running_losses = []
running_rewards = []

for exp_num in range(n_exp):
    seed = seeds[exp_num]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    agent_network = ModelPerceptron(dim_observation, dim_action, dim_hidden=dim_hidden, n_hidden_layers=n_hidden_layers)
    rl_agent = GaussianPDFModel(
        model=agent_network,
        dim_observation=dim_observation,
        dim_action=dim_action,
        action_bounds=np.array([[0, 1]]),
        scale_factor=scale_factor,
        std=std,
    )

    rl_critic = Critic(ModelPerceptron(dim_observation, 1, dim_hidden=dim_hidden_critic, n_hidden_layers=n_hidden_layers_critic))

    optimizer_agent = torch.optim.SGD(rl_agent.parameters(), lr=lr)

    running_loss = []
    running_reward = []
    for iter_num in tqdm(range(N_iter)):
        samples_ind = np.unique(rng.integers(0, n_poses, size=N_episode))
        samples_ind.sort()
        samples = pl_to_enter[samples_ind]
        place_ratios = np.zeros(len(samples))
        
        poses_true_info = run_true_simulation(init_lob, diffs_grouped, orders_per_diff, samples, place_ratios, rng)
        poses_pred_info, obs_actions = run_pred_simulation(init_lob, diffs_grouped, orders_per_diff, samples, place_ratios, rl_agent)
        #precalculate all costs and observation in needed format
        all_episod_records = []
        all_costs_td = []
        all_costs_per_step = []
        reward = 0
        M = 0
        for i, episod_record in enumerate(obs_actions):
            all_episod_records.append(torch.FloatTensor(episod_record))
            costs_per_step = torch.ones(len(episod_record)) * (2 * (poses_true_info[i] != poses_pred_info[i]) - 1) / len(episod_record)
            if len(episod_record) > N_td:
                all_costs_td.append((costs_per_step[:-1].unfold(0, N_td, 1) * gamma**torch.arange(0, N_td)).sum(dim=1))
            else:
                all_costs_td.append(torch.Tensor([]))
            all_costs_per_step.append(costs_per_step)
            if not ((poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == -1) or \
                    (poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == None)):
                if len(episod_record):
                    reward += poses_true_info[i] == poses_pred_info[i]
                    M += 1
        reward /= M

        all_episod_records = torch.concat(all_episod_records, dim=0)
        # critic train
        optimizer_critic = torch.optim.SGD(rl_critic.parameters(), lr=lr_critic)
        for _ in range(N_critic_epoch):
            #values = rl_critic(all_episod_records[:, :-1]) # last value - action
            running_idx = 0
            for i, episod_record in enumerate(obs_actions):
                if not ((poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == -1) or \
                        (poses_true_info[i] == poses_pred_info[i] and poses_true_info[i] == None)):
                    if len(episod_record) > N_td:
                        values = rl_critic(all_episod_records[running_idx:running_idx+len(episod_record), :-1]).reshape(-1, )
                        optimizer_critic.zero_grad()
                        loss = critic_objective_td(all_costs_td[i], 
                                                values,#values[running_idx:running_idx+len(episod_record)],
                                                gamma, N_td)
                        loss.backward()
                        optimizer_critic.step()
                        #print(loss, values)
                running_idx += len(episod_record)
        # agent train
        optimizer_agent.zero_grad()
        loss = actor_objective(all_episod_records, poses_true_info, poses_pred_info,
                            obs_actions, all_costs_per_step, rl_agent, rl_critic, gamma)
        loss.backward()
        optimizer_agent.step()
        # # PPO train
        # loss = 0
        # optimizer_agent = torch.optim.SGD(rl_agent.parameters(), lr=lr)
        # for _ in tqdm(range(N_ppo_epoch)):
        #     optimizer_agent.zero_grad()
        #     loss_epoch = ppo_objective(all_episod_records, poses_true_info, poses_pred_info,
        #                         obs_actions, all_costs_per_step, rl_agent, rl_critic, gamma, epsilon)
        #     loss_epoch.backward()
        #     optimizer_agent.step()
        #     loss += loss_epoch.detach()
        # loss /= N_ppo_epoch
        running_loss.append(loss.item())
        running_reward.append(reward)
    running_losses.append(running_loss)
    running_rewards.append(running_reward)

save_dir = f'./experiments/exp_critic{103}/'
with open(save_dir + 'hypers.pkl', 'wb') as f:
    pickle.dump(params, f)
with open(save_dir + 'seeds.npy', 'wb') as f:
    np.save(f, seeds)
with open(save_dir + 'losses.npy', 'wb') as f:
    np.save(f, running_losses)
with open(save_dir + 'rewards.npy', 'wb') as f:
    np.save(f, running_rewards)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for i, loss in enumerate(running_losses):
    sns.lineplot(loss, marker='.', ax=axes[0])
for i, reward in enumerate(running_rewards):
    sns.lineplot(reward, marker='.', ax=axes[1])
plt.savefig(save_dir + f'loss_reward_all')

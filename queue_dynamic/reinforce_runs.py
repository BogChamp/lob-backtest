import torch
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from queue_dynamic.simulation import run_true_simulation, run_pred_simulation
from queue_dynamic.losses import reinforce_objective, reinforce_objective_with_baseline
from queue_dynamic.models.models import GaussianPDFModel, ModelPerceptron
from lobio.utils.utils import group_diffs, group_orders

diffs_prepared_file = "./data/diffs_prepared.npy"
init_lob_prepared_file = "./data/init_lob_prepared.npy"
orders_prepared_file = "./data/orders_prepared.npy"
pl_to_enter_file = "./data/price_level_to_enter.npy"

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
dim_hidden = 8
n_hidden_layers = 1
std = 0.01
scale_factor = 10
gamma = 0.95
lr = 0.001
N_iter = 50
N_episode = 500
n_exp = 25

params = {'dim_obs': dim_observation, 
          'dim_act': dim_action, 
          'dim_hid': dim_hidden,
          'n_hid_layers': n_hidden_layers,
          'std': std, 
          'scale_factor': scale_factor,
          'gamma': gamma,
          'lr': lr,
          'N_iter': N_iter,
          'N_episod': N_episode,
          'n_exp': n_exp,
          'dynamic': 'zero'
          }

seeds = [int(round(np.random.uniform(0, 10), 5) * 10**5) for _ in range(n_exp)]
running_losses = []
running_rewards = []

for exp_num in range(n_exp):
    seed = seeds[exp_num]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    #seeds.append(seed)

    model_perc = ModelPerceptron(dim_observation, dim_action, dim_hidden=dim_hidden, n_hidden_layers=n_hidden_layers)
    rl_model = GaussianPDFModel(
        model=model_perc,
        dim_observation=dim_observation,
        dim_action=dim_action,
        action_bounds=np.array([[0, 1]]),
        scale_factor=scale_factor,
        std=std,
    )
    optimizer = torch.optim.SGD(rl_model.parameters(), lr=lr)

    running_loss = []
    running_reward = []
    baseline = torch.FloatTensor([])
    for i in tqdm(range(N_iter)):
        #samples = rng.choice(pl_to_enter, size=1000, replace=False, axis=0)
        samples_ind = np.unique(rng.integers(0, n_poses, size=N_episode))
        samples_ind.sort()
        samples = pl_to_enter[samples_ind]
        place_ratios = np.zeros(len(samples))#rng.uniform(0, 0.3, size=len(samples))#rng.triangular(0, 0.1, 1, size=len(samples))

        optimizer.zero_grad()
        poses_true_info = run_true_simulation(init_lob, diffs_grouped, orders_per_diff, samples, place_ratios, rng)
        poses_pred_info, obs_actions = run_pred_simulation(init_lob, diffs_grouped, orders_per_diff, samples, place_ratios, rl_model)
        loss, reward = reinforce_objective(poses_true_info, poses_pred_info, obs_actions, rl_model, gamma)
        # loss, reward, baseline = reinforce_objective_with_baseline(poses_true_info, poses_pred_info, obs_actions, rl_model, gamma, baseline)
        running_loss.append(loss.item())
        running_reward.append(reward)

        loss.backward()
        optimizer.step()

    running_losses.append(running_loss)
    running_rewards.append(running_reward)

save_dir = f'./experiments/exp{102}/'
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

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Tuple


class ModelPerceptron(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_hidden: int,
        n_hidden_layers: int,
        leaky_relu_coef: float = 0.15,
        is_bias: bool = True,
    ):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_hidden_layers = n_hidden_layers
        self.leaky_relu_coef = leaky_relu_coef
        self.is_bias = is_bias

        self.input_layer = nn.Linear(dim_input, dim_hidden, bias=is_bias)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(dim_hidden, dim_hidden, bias=is_bias)
                for _ in range(n_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(dim_hidden, dim_output, bias=is_bias)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = nn.functional.leaky_relu(
            self.input_layer(x), negative_slope=self.leaky_relu_coef
        )
        for layer in self.hidden_layers:
            x = nn.functional.leaky_relu(layer(x), negative_slope=self.leaky_relu_coef)
        x = self.output_layer(x)
        return x


class GaussianPDFModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dim_observation: int,
        dim_action: int,
        std: float,
        action_bounds: np.array,
        scale_factor: float,
    ):
        super().__init__()
        self.std = std
        self.model = model
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        self.scale_factor = scale_factor
        self.register_parameter(
            name="scale_tril_matrix",
            param=torch.nn.Parameter(
                (self.std * torch.eye(self.dim_action)).float(),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            name="action_bounds",
            param=torch.nn.Parameter(
                torch.tensor(action_bounds).float(),
                requires_grad=False,
            ),
        )

        # self.perceptron = nn.Sequential(nn.Linear(self.dim_observation, self.dim_hidden), nn.LeakyReLU(self.leakyrelu_coef),
        #                                 nn.Linear(self.dim_hidden, self.dim_hidden), nn.LeakyReLU(self.leakyrelu_coef),
        #                                 nn.Linear(self.dim_hidden, self.dim_action))
        #self.perceptron = nn.Sequential(nn.Linear(self.dim_observation, self.dim_action))

    def get_unscale_coefs_from_minus_one_one_to_action_bounds(
        self,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        action_bounds = self.get_parameter("action_bounds")
        beta_ = action_bounds.mean(dim=1)
        lambda_ = action_bounds[:, 1] - beta_

        return beta_, lambda_

    def unscale_from_minus_one_one_to_action_bounds(
        self, x: torch.FloatTensor
    ) -> torch.FloatTensor:
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return x * unscale_multiplier #+ unscale_bias

    def scale_from_action_bounds_to_minus_one_one(
        self, y: torch.FloatTensor
    ) -> torch.FloatTensor:
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return (y - unscale_bias) / unscale_multiplier

    def get_means(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        out = self.model(observations)
        return (1 - 3 * self.std) * torch.tanh(out / self.scale_factor)

    def split_to_observations_actions(
        self, observations_actions: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        observation, action = (
                observations_actions[:, : self.dim_observation],
                observations_actions[:, self.dim_observation :],
            )

        return observation, action

    def log_probs(self, batch_of_observations_actions: torch.FloatTensor) -> torch.FloatTensor:
        observations, actions = self.split_to_observations_actions(
            batch_of_observations_actions
        )

        scale_tril_matrix = self.get_parameter("scale_tril_matrix")
        scaled_mean = self.get_means(observations)
        scaled_action = self.scale_from_action_bounds_to_minus_one_one(actions)
        log_probs = MultivariateNormal(scaled_mean, scale_tril=scale_tril_matrix).log_prob(scaled_action)
        return log_probs

    def sample(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        action_bounds = self.get_parameter("action_bounds")
        scale_tril_matrix = self.get_parameter("scale_tril_matrix")
        scaled_mean = self.get_means(observation)
        #print(scaled_mean)
        sampled_scaled_action = MultivariateNormal(scaled_mean, scale_tril=scale_tril_matrix).sample()
        #print(sampled_scaled_action)
        sampled_action = self.unscale_from_minus_one_one_to_action_bounds(sampled_scaled_action)
        #print(sampled_action)

        return torch.clamp(
            sampled_action, action_bounds[:, 0], action_bounds[:, 1]
        )
import torch
from torch import nn


class Model(nn.Module):
    """
    Actor class -  acts as the policy, choosing actions based on the current state

    :param state_size: The length of the state vector
    :param action_size: The length of the action vector
    :param hidden_size: Number of nodes in the hidden layer
    """

    def __init__(self, state_size, action_size, hidden_size):
        super(Model, self).__init__()
        self.actor_base = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_size)
        self.std = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softplus()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        """
        Choose action for the current state

        :param state: The current state
        :return: Distribution of possible actions
        """
        base = self.actor_base(state)
        mean = self.mean(base)
        std = self.std(base)
        value = self.critic(state)
        return torch.distributions.Normal(mean, std), value


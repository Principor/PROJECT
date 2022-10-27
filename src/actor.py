import torch
from torch import nn


class Actor(nn.Module):
    """
    Actor class -  acts as the policy, choosing actions based on the current state

    :param state_size: The length of the state vector
    :param action_size: The length of the action vector
    :param hidden_size: Number of nodes in the hidden layer
    """

    def __init__(self, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_size)
        self.std = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softplus()
        )

    def forward(self, state):
        """
        Choose action for the current state

        :param state: The current state
        :return: Distribution of possible actions
        """
        base = self.base(state)
        mean = self.mean(base)
        std = self.std(base)
        return torch.distributions.Normal(mean, std)


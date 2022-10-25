import torch
from torch import nn


class Actor(nn.Module):
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
        base = self.base(state)
        mean = self.mean(base)
        std = self.std(base)
        return torch.distributions.Normal(mean, std)


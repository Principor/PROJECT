import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return torch.distributions.Categorical(self.actor(state))

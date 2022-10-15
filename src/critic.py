from torch import nn


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_size, 1),
        )

    def forward(self, state):
        return self.critic(state)

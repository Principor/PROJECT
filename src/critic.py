from torch import nn


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 1),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        return self.actor(state)

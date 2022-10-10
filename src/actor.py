from torch import nn


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, action_size),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        return self.actor(state)

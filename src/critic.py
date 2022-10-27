from torch import nn


class Critic(nn.Module):
    """
    Critic class - acts as a value function, tries to learn the value of each state in order to calculate the advantage

    :param state_size: The length of the state vector
    :param hidden_size: Number of nodes in the hidden layer
    """
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        """
        Predict the value of the current state

        :param state: The current state
        :return: The predicted value
        """
        return self.critic(state)

import torch
from torch import nn


HIDDEN_SIZE = 128


def state_to_tensor(state):
    return torch.unsqueeze(torch.tensor(state), 0)


class Model(nn.Module):
    """
    Model class, contains actor network and critic network for generating actions and evaluating values respectively

    :param state_size: The length of the state vector
    :param action_size: The length of the action vector
    """

    def __init__(self, state_size, action_size):
        super(Model, self).__init__()

        self.actor_lstm = nn.LSTM(state_size, HIDDEN_SIZE, 1)
        self.actor_base = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.mean = nn.Linear(HIDDEN_SIZE, action_size)
        self.log_std = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size)
        )

        self.critic_lstm = nn.LSTM(state_size, HIDDEN_SIZE, 1)
        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

        self.actor_state, self.critic_state = None, None

    def initialise_lstm_states(self, batch_size):
        self.actor_state = (torch.zeros((1, batch_size, HIDDEN_SIZE)), torch.zeros((1, batch_size, HIDDEN_SIZE)))
        self.critic_state = (torch.zeros((1, batch_size, HIDDEN_SIZE)), torch.zeros((1, batch_size, HIDDEN_SIZE)))

    def apply_mask(self, mask):
        mask = torch.tensor(mask, dtype=torch.float32)

        actor_hidden, actor_cell = self.actor_state
        critic_hidden, critic_cell = self.critic_state

        self.actor_state = (actor_hidden * mask, actor_cell * mask)
        self.critic_state = (critic_hidden * mask, critic_cell * mask)

    def forward(self, state):
        """
        Generate action distribution and evaluate value for given state

        :param state: The current state
        :return: Distribution of possible actions, value of current state
        """
        actor_lstm, self.actor_state = self.actor_lstm(state, self.actor_state)
        base = self.actor_base(actor_lstm)
        mean = self.mean(base)
        std = self.log_std(base).exp()

        critic_lstm, self.critic_state = self.critic_lstm(state, self.critic_state)
        value = self.critic(critic_lstm)

        return torch.distributions.Normal(mean, std), value


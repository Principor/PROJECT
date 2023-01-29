import torch
from torch import nn


def state_to_tensor(state):
    return torch.tensor(state).unsqueeze(0)


class Model(nn.Module):
    """
    Model class, contains actor network and critic network for generating actions and evaluating values respectively

    :param state_size: The length of the state vector
    :param action_size: The length of the action vector
    :param hidden_size: Number of nodes in the hidden layer
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super(Model, self).__init__()

        self.hidden_size = hidden_size

        self.actor_network = nn.Sequential(
            nn.Linear(state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.critic_network = nn.Sequential(
            nn.Linear(state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )

        self.actor_lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.critic_lstm = nn.LSTM(self.hidden_size, self.hidden_size)

        self.mean = nn.Linear(self.hidden_size, action_size)
        self.log_std = nn.Linear(self.hidden_size, action_size)
        self.value = nn.Linear(self.hidden_size, 1)

        self.actor_lstm_state = self.critic_lstm_state = None

    def initialise_hidden_states(self, batch_size):
        self.actor_lstm_state = (torch.zeros(1, batch_size, self.hidden_size),
                                 torch.zeros(1, batch_size, self.hidden_size))
        self.critic_lstm_state = (torch.zeros(1, batch_size, self.hidden_size),
                                  torch.zeros(1, batch_size, self.hidden_size))

    def apply_mask(self, mask):
        self.actor_lstm_state = [item * mask for item in self.actor_lstm_state]
        self.critic_lstm_state = [item * mask for item in self.critic_lstm_state]

    def forward(self, state, dones=None):
        """
        Generate action distribution and evaluate value for given state

        :param state: The current state
        :return: Distribution of possible actions, value of current state
        """
        sequence_length, buffer_size = state.shape[:2]

        actor_forward = self.actor_network(state)
        critic_forward = self.critic_network(state)

        actor_lstm = torch.zeros(sequence_length, buffer_size, self.hidden_size)
        critic_lstm = torch.zeros(sequence_length, buffer_size, self.hidden_size)
        for i in range(sequence_length):
            actor_lstm_in = actor_forward[i].unsqueeze(0)
            critic_lstm_in = critic_forward[i].unsqueeze(0)
            actor_lstm_out, self.actor_lstm_state = self.actor_lstm(actor_lstm_in, self.actor_lstm_state)
            critic_lstm_out, self.critic_lstm_state = self.critic_lstm(critic_lstm_in, self.critic_lstm_state)
            actor_lstm[i] = actor_lstm_out.squeeze(0)
            critic_lstm[i] = critic_lstm_out.squeeze(0)
            if dones is not None:
                self.apply_mask(1 - dones[i])

        mean = self.mean(actor_lstm)
        std = self.log_std(actor_lstm).exp()
        value = self.value(critic_lstm)
        return torch.distributions.Normal(mean, std), value

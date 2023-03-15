import enum

import torch
from torch import nn


def state_to_tensor(state):
    return torch.tensor(state).unsqueeze(0)


class StateMaskType(enum.Enum):
    NO_STATE_MASK = 0
    ACTOR_STATE_MASK = 1
    FULL_STATE_MASK = 2


class Model(nn.Module):
    """
    Model class, contains actor network and critic network for generating actions and evaluating values respectively

    :param state_size: The length of the state vector
    :param action_size: The length of the action vector
    :param hidden_size: Number of nodes in the hidden layer
    """

    def __init__(self, state_size, action_size, recurrent_layers=False,
                 state_mask_type=StateMaskType.NO_STATE_MASK):
        super(Model, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = 128
        self.recurrent_layers = recurrent_layers
        self.state_mask_type = state_mask_type

        self.state_mask = torch.ones(21)
        self.state_mask[-5:] = 0

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
        self.log_std = nn.Parameter(torch.zeros(action_size))
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

        actor_state = state
        critic_state = state
        if self.state_mask_type == StateMaskType.ACTOR_STATE_MASK:
            actor_state *= self.state_mask
        elif self.state_mask_type == StateMaskType.FULL_STATE_MASK:
            actor_state *= self.state_mask
            critic_state *= self.state_mask

        actor_forward = self.actor_network(actor_state)
        critic_forward = self.critic_network(critic_state)

        actor_lstm = torch.zeros_like(actor_forward)
        critic_lstm = torch.zeros_like(critic_forward)
        if self.recurrent_layers:
            for i in range(sequence_length):
                actor_lstm_in = actor_forward[i].unsqueeze(0)
                critic_lstm_in = critic_forward[i].unsqueeze(0)
                actor_lstm_out, self.actor_lstm_state = self.actor_lstm(actor_lstm_in, self.actor_lstm_state)
                critic_lstm_out, self.critic_lstm_state = self.critic_lstm(critic_lstm_in, self.critic_lstm_state)
                actor_lstm[i] = actor_lstm_out.squeeze(0)
                critic_lstm[i] = critic_lstm_out.squeeze(0)
                if dones is not None:
                    self.apply_mask(1 - dones[i])
        else:
            actor_lstm = actor_forward
            critic_lstm = critic_forward

        mean = self.mean(actor_lstm)
        std = self.log_std.exp()
        value = self.value(critic_lstm)
        return torch.distributions.Normal(mean, std), value

    def save_model(self, path):
        torch.save({
            'state_size': self.state_size,
            'action_size': self.action_size,
            'state_dict': self.state_dict(),
            'recurrent_layers': self.recurrent_layers,
            'state_mask_type': self.state_mask_type
        }, path)

    @staticmethod
    def load_model(path):
        checkpoint = torch.load(path)
        state_size = checkpoint['state_size']
        action_size = checkpoint['action_size']
        recurrent_layers = checkpoint['recurrent_layers']
        state_mask_type = checkpoint['state_mask_type']
        model = Model(state_size, action_size, recurrent_layers, state_mask_type)
        model.load_state_dict(checkpoint['state_dict'])
        return model

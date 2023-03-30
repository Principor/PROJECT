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
    :param recurrent_layers: Whether recurrent layers should be used
    :param state_mask_type: Which networks should have their inputs masked
    """

    def __init__(self, state_size, action_size, recurrent_layers=False,
                 state_mask_type=StateMaskType.NO_STATE_MASK):
        super(Model, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = 128
        self.recurrent_layers = recurrent_layers
        self.state_mask_type = state_mask_type

        self.state_mask = torch.ones(state_size)
        self.state_mask[15:] = 0

        self.actor_0 = nn.Sequential(nn.Linear(state_size, self.hidden_size), nn.ReLU())
        self.critic_0 = nn.Sequential(nn.Linear(state_size, self.hidden_size), nn.ReLU())
        
        self.actor_1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())
        self.critic_1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())

        self.actor_lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.critic_lstm = nn.LSTM(self.hidden_size, self.hidden_size)

        self.mean = nn.Linear(self.hidden_size, action_size)
        self.log_std = nn.Linear(self.hidden_size, action_size)
        self.value = nn.Linear(self.hidden_size, 1)

        self.actor_lstm_state = self.critic_lstm_state = None

    def initialise_hidden_states(self, batch_size):
        """
        Initialise LSTM memory

        :param batch_size: The batch size that will be used during training
        """
        self.actor_lstm_state = (torch.zeros(1, batch_size, self.hidden_size),
                                 torch.zeros(1, batch_size, self.hidden_size))
        self.critic_lstm_state = (torch.zeros(1, batch_size, self.hidden_size),
                                  torch.zeros(1, batch_size, self.hidden_size))

    def apply_mask(self, mask):
        """
        Apply a mask to reset memory for finished episodes

        :param mask: The mask to apply
        """
        self.actor_lstm_state = [item * mask for item in self.actor_lstm_state]
        self.critic_lstm_state = [item * mask for item in self.critic_lstm_state]

    def forward(self, state, dones=None):
        """
        Generate action distribution and evaluate value for given state

        :param state: The current state
        :param dones: Whether each state was terminal
        :return: Distribution of possible actions, value of current state
        """
        sequence_length, _ = state.shape[:2]

        actor_state = state
        critic_state = state
        if self.state_mask_type == StateMaskType.ACTOR_STATE_MASK:
            actor_state *= self.state_mask
        elif self.state_mask_type == StateMaskType.FULL_STATE_MASK:
            actor_state *= self.state_mask
            critic_state *= self.state_mask

        actor_0 = self.actor_0(actor_state)
        critic_0 = self.critic_0(critic_state)

        actor_1 = torch.zeros_like(actor_0)
        critic_1 = torch.zeros_like(critic_0)
        if self.recurrent_layers:
            for i in range(sequence_length):
                actor_in = actor_0[i].unsqueeze(0)
                critic_in = critic_0[i].unsqueeze(0)
                actor_out, self.actor_lstm_state = self.actor_lstm(actor_in, self.actor_lstm_state)
                critic_out, self.critic_lstm_state = self.critic_lstm(critic_in, self.critic_lstm_state)
                actor_1[i] = actor_out.squeeze(0)
                critic_1[i] = critic_out.squeeze(0)
                if dones is not None:
                    self.apply_mask(1 - dones[i])
        else:
            actor_1 = self.actor_1(actor_0)
            critic_1 = self.critic_1(critic_0)

        mean = self.mean(actor_1)
        std = self.log_std(actor_1).exp()
        value = self.value(critic_1)
        return torch.distributions.Normal(mean, std), value

    def save_model(self, path):
        """
        Save a model

        :param name: The name to give the saved model
        """
        torch.save({
            'state_size': self.state_size,
            'action_size': self.action_size,
            'state_dict': self.state_dict(),
            'recurrent_layers': self.recurrent_layers,
            'state_mask_type': self.state_mask_type
        }, path)

    @staticmethod
    def load_model(path):
        """
        Load a model

        :param name: The name of the model to load
        :return: The loaded model
        """
        checkpoint = torch.load(path)
        state_size = checkpoint['state_size']
        action_size = checkpoint['action_size']
        recurrent_layers = checkpoint['recurrent_layers']
        state_mask_type = checkpoint['state_mask_type']
        model = Model(state_size, action_size, recurrent_layers, state_mask_type)
        model.load_state_dict(checkpoint['state_dict'])
        return model

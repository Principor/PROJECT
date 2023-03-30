import os
import random
import time

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from model import Model, StateMaskType, state_to_tensor
import racecar_driving

# Parameters
NUM_UPDATES = 50
NUM_ENVS = 4
NUM_STEPS = 8192
BATCH_SIZE = 2048
SEQUENCE_LENGTH = 16

NUM_EPOCHS = 10
EPSILON = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
CRITIC_DISCOUNT = 0.5
LEARNING_RATE = 0.0003
DECAY_LR = True
MAX_GRAD_NORM = 0.5

LOG_FREQUENCY = 5
RUN_NAME = "lstm_asymmetric"
RECURRENT_LAYERS = True
STATE_MASK_TYPE = StateMaskType.ACTOR_STATE_MASK
CAR_INDEX = -1


def normalise(x):
    """
    Normalise an array to have a mean of 0 and standard deviation of 1

    :param x: The array to be normalised
    :return: Normalised version of x
    """
    x -= x.mean()
    x /= (x.std() + 1e-8)   # Avoid divide by 0 error
    return x


class Agent:
    """
    Agent class, runs the Proximal Policy Algorithm
    Paper: https://arxiv.org/abs/1707.06347

    :param state_size: The length of the state vector
    :param action_size: The length of the action vector
    :param hidden_size: Number of nodes in the hidden layer

    :param num_epochs: The number of updates to complete at each step
    :param epsilon: Clipping parameter
    :param gamma: Discount factor
    :param lr: Learning rate
    """
    def __init__(self, state_size, action_size, recurrent_layers, state_mask_type, num_updates, num_envs, batch_size,
                 sequence_length, num_epochs, epsilon, gamma, gae_lambda, critic_discount, lr, decay_lr, max_grad_norm):
        # Create models
        self.model = Model(state_size, action_size, recurrent_layers, state_mask_type)
        self.model.initialise_hidden_states(num_envs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Store parameters
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_epochs = num_epochs
        self.eps = epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.critic_discount = critic_discount
        self.lr = lr
        self.decay_lr = decay_lr
        self.max_grad_norm = max_grad_norm

        self.updates_completed = 0

        # Initialise memory
        self.state_memory = []
        self.action_memory = []
        self.prob_memory = []
        self.value_memory = []
        self.reward_memory = []
        self.terminated_memory = []

        self.actor_hidden_memory = []
        self.actor_cell_memory = []
        self.critic_hidden_memory = []
        self.critic_cell_memory = []

    def choose_action(self, state):
        """
        Choose an action for the current state, and remember the state, action, log probability of the action and
        estimated value.

        :param state: The current state
        :return: Chosen action vector
        """
        self.actor_hidden_memory.append(self.model.actor_lstm_state[0])
        self.actor_cell_memory.append(self.model.actor_lstm_state[1])
        self.critic_hidden_memory.append(self.model.critic_lstm_state[0])
        self.critic_cell_memory.append(self.model.critic_lstm_state[1])

        # Generate action
        with torch.no_grad():
            state = state_to_tensor(state)
            dist, value = self.model(state)
            action = dist.sample()
            prob = dist.log_prob(action)

        # Store information
        self.state_memory.append(state.squeeze(0))
        self.action_memory.append(action.numpy().squeeze(0))
        self.prob_memory.append(prob.numpy().squeeze(0))
        self.value_memory.append(value.numpy().squeeze(0))

        return action.detach().numpy().squeeze(0)

    def finish_step(self, reward, terminated):
        """
        Remember the reward and whether a terminal state was reached

        :param reward: Reward at the current step
        :param terminated: Whether the current step reach a terminal state
        """
        self.reward_memory.append(np.expand_dims(reward, -1).astype(np.float32))
        self.terminated_memory.append(np.expand_dims(terminated, -1).astype(np.float32))

        self.model.apply_mask(torch.tensor(1 - self.terminated_memory[-1]))

    def calculate_returns(self, final_value):
        """
        Calculate the discounted returns from the experienced rewards
        This uses Generalised Advantage Estimation in order to reduce variance in the returns

        :param final_value: The predicted value of the next state
        :return: The list of returns from each step
        """
        length = len(self.reward_memory)
        next_values = self.value_memory[1:] + [final_value.detach().numpy()]
        returns = []
        gae = 0
        for step in reversed(range(length)):
            mask = 1 - self.terminated_memory[step]
            delta = self.reward_memory[step] + self.gamma * next_values[step] * mask - self.value_memory[step]
            gae = delta + self.gae_lambda * self.gamma * gae * mask
            returns.insert(0, gae + self.value_memory[step])
        return np.stack(returns, axis=1)

    def learn(self, next_state):
        """
        Update models based on the sampled experience

        :param next_state: The next state when training was stopped. Used to predict returns that would follow after
        cut-off point
        """
        old_actor_state, old_critic_state = self.model.actor_lstm_state, self.model.critic_lstm_state
        final_value = self.model(state_to_tensor(next_state))[1].squeeze(0)
        returns = self.calculate_returns(final_value)
        buffer_size = returns.size
        num_sequences = buffer_size // self.sequence_length

        # Stack and reshape all sequences
        sequences = (
            returns,
            normalise(returns - np.stack(self.value_memory, axis=1)),
            np.stack(self.state_memory, axis=1),
            np.stack(self.action_memory, axis=1),
            np.stack(self.prob_memory, axis=1),
            np.stack(self.terminated_memory, axis=1)
        )
        sequences = (np.reshape(sequence, (num_sequences, self.sequence_length, -1)) for sequence in sequences)
        sequences = (np.moveaxis(sequence, 0, 1) for sequence in sequences)
        all_returns, all_advantages, all_states, all_actions, all_probs, all_dones = sequences

        lstm_states = (
            np.stack(self.actor_hidden_memory, axis=2),
            np.stack(self.actor_cell_memory, axis=2),
            np.stack(self.critic_hidden_memory, axis=2),
            np.stack(self.critic_cell_memory, axis=2)
        )
        lstm_state_shape = (1, num_sequences, self.sequence_length, self.model.hidden_size)
        lstm_states = (np.reshape(lstm_state, lstm_state_shape)[:, :, 0] for lstm_state in lstm_states)
        actor_hidden_states, actor_cell_states, critic_hidden_states, critic_cell_states = lstm_states

        # Decay learning rate
        if self.decay_lr:
            self.optimizer.param_groups[0]['lr'] = self.lr * (1 - self.updates_completed / self.num_updates)
        self.updates_completed += 1

        for _ in range(self.num_epochs):

            # Generate indices for each batch
            indices = np.arange(num_sequences)
            np.random.shuffle(indices)
            indices = np.split(indices, buffer_size // self.batch_size)

            # Create each batch
            batches = [(
                torch.tensor(all_returns[:, batch_indices]),
                torch.tensor(all_advantages[:, batch_indices]),
                torch.tensor(all_states[:, batch_indices]),
                torch.tensor(all_actions[:, batch_indices]),
                torch.tensor(all_probs[:, batch_indices]),
                torch.tensor(all_dones[:, batch_indices]),
                (
                    torch.tensor(actor_hidden_states[:, batch_indices]),
                    torch.tensor(actor_cell_states[:, batch_indices]),
                ),
                (
                    torch.tensor(critic_hidden_states[:, batch_indices]),
                    torch.tensor(critic_cell_states[:, batch_indices]),
                ),
            ) for batch_indices in indices]

            for batch in batches:
                returns, advantages, states, actions, old_probs, dones, actor_states, critic_states = batch

                # Get current distribution and value from models
                self.model.actor_lstm_state, self.model.critic_lstm_state = actor_states, critic_states
                dist, new_values = self.model(states, dones)
                new_values = new_values

                # Calculate components of actor loss
                new_probs = dist.log_prob(actions)
                ratios = torch.exp(new_probs - old_probs)

                unclipped = ratios * advantages
                clipped = torch.clip(ratios, 1-self.eps, 1+self.eps) * advantages

                # Calculate actual loss
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = torch.nn.functional.mse_loss(returns, new_values).mean()
                loss = actor_loss + self.critic_discount * critic_loss

                # Back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Clear all memory
        self.state_memory.clear()
        self.action_memory.clear()
        self.prob_memory.clear()
        self.value_memory.clear()
        self.reward_memory.clear()
        self.terminated_memory.clear()

        self.actor_hidden_memory.clear()
        self.actor_cell_memory.clear()
        self.critic_hidden_memory.clear()
        self.critic_cell_memory.clear()

        self.model.actor_lstm_state, self.model.critic_lstm_state = old_actor_state, old_critic_state

    def save_model(self):
        """
        Save the model
        """
        path = "../models/{}".format(RUN_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_model(path + "/model.pth")


def make_env():
    track_list = None
    return gym.make('RacecarDriving-v0', car_index=CAR_INDEX, track_list=track_list, transform_tracks=False)


def train():
    """
    Train the model
    """

    # Vectorise and wrap environment
    envs = SubprocVecEnv([lambda: make_env() for _ in range(NUM_ENVS)])
    envs = VecMonitor(envs)
    envs = VecNormalize(envs, gamma=GAMMA)
    print()
    print(RUN_NAME)

    writer = SummaryWriter("../summaries/" + RUN_NAME)

    agent = Agent(envs.observation_space.shape[0], envs.action_space.shape[0], RECURRENT_LAYERS, STATE_MASK_TYPE,
                  NUM_UPDATES, NUM_ENVS, BATCH_SIZE, SEQUENCE_LENGTH, NUM_EPOCHS, EPSILON, GAMMA, GAE_LAMBDA,
                  CRITIC_DISCOUNT, LEARNING_RATE, DECAY_LR, MAX_GRAD_NORM)

    # Save the score of each episode to track progress
    scores = []
    start_time = time.time()

    observation = envs.reset()
    for update in range(NUM_UPDATES):
        for update_step in range(NUM_STEPS):
            action = agent.choose_action(observation)
            observation, reward, done, info = envs.step(action)
            agent.finish_step(reward, done)

            # Record the return of each finished episode
            for i in range(NUM_ENVS):
                item = info[i]
                if 'episode' in item.keys():
                    score = item['episode']['r']
                    scores.append(score)
                    writer.add_scalar('score', score, (update * NUM_STEPS + update_step) * NUM_ENVS + i)

        # Update models
        agent.learn(observation)

        # Output progress
        if (update + 1) % LOG_FREQUENCY == 0:
            print("Update: {}".format(update).ljust(15), end="")
            print("Score: {}".format(round(float(np.mean(scores)), 2)).ljust(20), end="")
            print("Time: {}".format(round(time.time() - start_time, 2)))
            scores.clear()

    agent.save_model()
    envs.save("../models/{}/normaliser".format(RUN_NAME))   # Save normaliser
    envs.close()
    writer.close()


if __name__ == '__main__':
    train()

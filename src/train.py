import os
import time

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from model import Model
import racecar_driving

# Parameters
NUM_UPDATES = 100
NUM_ENVS = 4
NUM_STEPS = 512
BATCH_SIZE = 64

NUM_EPOCHS = 10
EPSILON = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
CRITIC_DISCOUNT = 0.5
LEARNING_RATE = 0.0003
DECAY_LR = True
MAX_GRAD_NORM = 0.5

HIDDEN_SIZE = 128

LOG_FREQUENCY = 10
RUN_NAME = "ppo"


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
    def __init__(self, state_size, action_size, hidden_size, num_updates, batch_size, num_epochs, epsilon, gamma,
                 gae_lambda, critic_discount, lr, decay_lr, max_grad_norm):
        # Create models
        self.model = Model(state_size, action_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Store parameters
        self.num_updates = num_updates
        self.batch_size = batch_size
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

    def choose_action(self, state):
        """
        Choose an action for the current state, and remember the state, action, log probability of the action and
        estimated value.

        :param state: The current state
        :return: Chosen action vector
        """

        # Generate action
        with torch.no_grad():
            state = torch.tensor(state)
            dist, value = self.model(state)
            action = dist.sample()
            prob = dist.log_prob(action)

        # Store information
        self.state_memory.append(state)
        self.action_memory.append(action.numpy())
        self.prob_memory.append(prob.numpy())
        self.value_memory.append(value.numpy().squeeze())

        return action.detach().numpy()

    def remember(self, reward, terminated):
        """
        Remember the reward and whether a terminal state was reached

        :param reward: Reward at the current step
        :param terminated: Whether the current step reach a terminal state
        """
        self.reward_memory.append(reward)
        self.terminated_memory.append(terminated)

    def calculate_returns(self, final_value):
        """
        Calculate the discounted returns from the experienced rewards
        This uses Generalised Advantage Estimation in order to reduce variance in the returns

        :param final_value: The predicted value of the next state
        :return: The list of returns from each step
        """
        length = len(self.reward_memory)
        next_values = self.value_memory[1:] + [final_value.detach().numpy().squeeze()]
        returns = []
        gae = 0
        for step in reversed(range(length)):
            mask = 1 - self.terminated_memory[step]
            delta = self.reward_memory[step] + self.gamma * next_values[step] * mask - self.value_memory[step]
            gae = delta + self.gae_lambda * self.gamma * gae * mask
            returns.insert(0, gae + self.value_memory[step])
        return np.stack(returns)

    def learn(self, next_state):
        """
        Update models based on the sampled experience

        :param next_state: The next state when training was stopped. Used to predict returns that would follow after
        cut-off point
        """

        returns = self.calculate_returns(self.model(torch.tensor(next_state))[1])

        # Stack and reshape all sequences
        sequences = (
            returns,
            normalise(returns - np.stack(self.value_memory)),
            np.stack(self.state_memory),
            np.stack(self.action_memory),
            np.stack(self.prob_memory)
        )
        sequences = (np.concatenate(sequence, axis=0) for sequence in sequences)
        all_returns, all_advantages, all_states, all_actions, all_probs = sequences

        # Decay learning rate
        if self.decay_lr:
            self.optimizer.param_groups[0]['lr'] = self.lr * (1 - self.updates_completed / self.num_updates)
        self.updates_completed += 1

        for _ in range(self.num_epochs):

            # Generate indices for each batch
            size = all_returns.shape[0]
            indices = np.arange(size)
            indices = np.split(indices, all_returns.shape[0] // self.batch_size)
            np.random.shuffle(indices)

            # Create each batch
            batches = [(
                torch.tensor(all_returns[batch_indices], dtype=torch.float32),
                torch.tensor(all_advantages[batch_indices], dtype=torch.float32),
                torch.tensor(all_states[batch_indices]),
                torch.tensor(all_actions[batch_indices]),
                torch.tensor(all_probs[batch_indices])
            ) for batch_indices in indices]

            for batch in batches:
                returns, advantages, states, actions, old_probs = batch

                advantages = torch.unsqueeze(advantages, dim=-1)

                # Get current distribution and value from models
                dist, new_values = self.model(states)
                new_values = torch.squeeze(new_values)

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

    def save_model(self):
        """
        Save the model
        """
        path = "../models/{}".format(RUN_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), path + "/model.pth")


def train():
    """
    Train the model
    """

    # Vectorise and wrap environment
    envs = SubprocVecEnv([lambda: gym.make('LunarLanderContinuous-v2') for _ in range(NUM_ENVS)])
    envs = VecMonitor(envs)
    envs = VecNormalize(envs, gamma=GAMMA)
    print()

    writer = SummaryWriter("../summaries/" + RUN_NAME)

    agent = Agent(envs.observation_space.shape[0], envs.action_space.shape[0], HIDDEN_SIZE, NUM_UPDATES, BATCH_SIZE,
                  NUM_EPOCHS, EPSILON, GAMMA, GAE_LAMBDA, CRITIC_DISCOUNT, LEARNING_RATE, DECAY_LR, MAX_GRAD_NORM)

    # Save the score of each episode to track progress
    scores = []

    observation = envs.reset()
    for update in range(NUM_UPDATES):
        for update_step in range(NUM_STEPS):
            action = agent.choose_action(observation)
            observation, reward, done, info = envs.step(action)
            agent.remember(reward, done)

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
            print("Update: {}\tAvg. score: {}".format(update, np.mean(scores)))
            scores.clear()

    envs.save("../models/normaliser")   # Save normaliser
    envs.close()
    writer.close()
    agent.save_model()


if __name__ == '__main__':
    start_time = time.time()
    train()
    print("Time taken: {}".format(time.time() - start_time))

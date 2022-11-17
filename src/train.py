import os
import time

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from critic import Critic

# Parameters
NUM_UPDATES = 100
NUM_ENVS = 4
NUM_STEPS = 1024
BATCH_SIZE = 512

NUM_EPOCHS = 20
EPSILON = 0.2
GAMMA = 0.99
LEARNING_RATE = 0.0003

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


def prepare_state(state):
    """
    Convert array into tensor

    :param state: Array representation of state
    :return: Tensor representation of state
    """
    return torch.tensor(state)


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
    def __init__(self, state_size, action_size, hidden_size, num_epochs, epsilon, gamma, lr):
        # Create models
        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Store parameters
        self.num_epochs = num_epochs
        self.eps = epsilon
        self.gamma = gamma

        # Initialise memory
        self.state_memory = []
        self.action_memory = []
        self.prob_memory = []
        self.reward_memory = []
        self.terminated_memory = []

    def choose_action(self, state):
        """
        Choose an action for the current state, and remember the state, action, and log probability of the action under
        the current policy.

        :param state: The current state
        :return: Chosen action vector
        """

        # Generate action
        with torch.no_grad():
            state = prepare_state(state)
            dist = self.actor(state)
            action = dist.sample()
            prob = dist.log_prob(action)

        # Store information
        self.state_memory.append(state)
        self.action_memory.append(action.numpy())
        self.prob_memory.append(prob.numpy())

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

        :param final_value: The predicted value of the next state
        :return: The list of returns from each step
        """
        current_return = final_value.detach().numpy().squeeze()
        returns = []
        for reward, terminated in reversed(list(zip(self.reward_memory, self.terminated_memory))):
            current_return = reward + self.gamma * current_return * (1 - terminated)
            returns.insert(0, current_return)   # Prepend as returns are calculated backwards
        returns = normalise(np.array(returns))
        return returns

    def learn(self, next_state):
        """
        Update models based on the sampled experience

        :param next_state: The next state when training was stopped. Used to predict returns that would follow after
        cut-off point
        """

        # Collect each sequence into a numpy array
        sequences = (
            self.calculate_returns(self.critic(torch.tensor(next_state))),
            np.stack(self.state_memory),
            np.stack(self.action_memory),
            np.stack(self.prob_memory)
        )
        sequences = (np.concatenate(sequence, axis=0) for sequence in sequences)
        all_returns, all_states, all_actions, all_probs = sequences

        for _ in range(self.num_epochs):

            # Generate indices for each batch
            size = all_returns.shape[0]
            indices = np.arange(size)
            np.random.shuffle(indices)
            indices = np.split(indices, all_returns.shape[0] // BATCH_SIZE)

            # Create each batch
            batches = [(
                torch.tensor(all_returns[batch_indices], dtype=torch.float32),
                torch.tensor(all_states[batch_indices]),
                torch.tensor(all_actions[batch_indices]),
                torch.tensor(all_probs[batch_indices])
            ) for batch_indices in indices]

            for batch in batches:
                returns, states, actions, old_probs = batch

                # Get current distribution and value from models
                dist = self.actor(states)
                new_values = torch.squeeze(self.critic(states))

                # Calculate components of actor loss
                new_probs = dist.log_prob(actions)
                ratios = torch.exp(new_probs - old_probs)
                advantages = torch.unsqueeze(returns - new_values.detach(), dim=-1)
                unclipped = ratios * advantages
                clipped = torch.clip(ratios, 1-self.eps, 1+self.eps) * advantages

                # Calculate actual loss
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = torch.nn.functional.mse_loss(returns, new_values).mean()
                loss = actor_loss + critic_loss

                # Back propagation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # Clear all memory
        self.state_memory.clear()
        self.action_memory.clear()
        self.prob_memory.clear()
        self.reward_memory.clear()
        self.terminated_memory.clear()

    def save_model(self):
        """
        Save the model
        """
        path = "../models/{}".format(RUN_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), path + "/actor.pth")


def train():
    """
    Train the model
    """
    envs = gym.vector.make("LunarLanderContinuous-v2", num_envs=NUM_ENVS)
    writer = SummaryWriter("../summaries/" + RUN_NAME)
    agent = Agent(envs.observation_space.shape[1], envs.action_space.shape[1], HIDDEN_SIZE, NUM_EPOCHS, EPSILON, GAMMA,
                  LEARNING_RATE)

    # Save the score of each episode to track progress
    saved_scores = []
    current_scores = np.zeros(NUM_ENVS)

    observation, info = envs.reset()
    for update in range(NUM_UPDATES):
        for update_step in range(NUM_STEPS):
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = envs.step(action)
            done = np.logical_or(terminated, truncated)
            current_scores += reward
            agent.remember(reward, done)

            for i in range(NUM_ENVS):
                if done[i]:
                    saved_scores.append(current_scores[i])

            current_scores *= 1 - done

        # Update models
        agent.learn(observation)

        # Output progress
        if (update + 1) % LOG_FREQUENCY == 0:
            print("Update: {}\tAvg. score: {}".format(update, np.mean(saved_scores)))
            saved_scores.clear()

    envs.close()
    writer.close()
    agent.save_model()


if __name__ == '__main__':
    start_time = time.time()
    train()
    print("Time taken: {}".format(time.time() - start_time))

import os

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from critic import Critic

NUM_UPDATES = 200
BUFFER_SIZE = 5000

NUM_EPOCHS = 20
GAMMA = 0.99
LEARNING_RATE = 0.001

LOG_FREQUENCY = 10
RUN_NAME = "actor_critic"


class Agent:
    def __init__(self, state_size, action_size, num_epochs, gamma, lr):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.gamma = gamma

        self.state_memory = []
        self.action_memory = []
        self.prob_memory = []
        self.reward_memory = []
        self.terminated_memory = []

    def choose_action(self, state):
        state_tensor = torch.tensor(state)
        dist = torch.distributions.Categorical(self.actor(state_tensor))
        action = dist.sample()
        self.state_memory.append(state)
        self.action_memory.append(action.detach().numpy())
        self.prob_memory.append(dist.log_prob(action).detach().numpy())
        return action.detach().numpy()

    def remember(self, reward, terminated):
        self.reward_memory.append(reward)
        self.terminated_memory.append(terminated)

    def learn(self, next_state):
        current_return = self.critic(torch.tensor(next_state)).item()
        returns = []
        for reward, terminated in reversed(list(zip(self.reward_memory, self.terminated_memory))):
            current_return = reward + self.gamma * current_return * (1 - terminated)
            returns.insert(0, current_return)
        returns = np.array(returns)
        returns = (returns - returns.std()) / returns.std()

        returns = torch.tensor(returns, dtype=torch.float32)
        states = torch.tensor(np.stack(self.state_memory))
        actions = torch.tensor(np.stack(self.action_memory))
        old_probs = torch.tensor(np.stack(self.prob_memory))

        for _ in range(self.num_epochs):
            dist = torch.distributions.Categorical(self.actor(states))
            new_values = torch.squeeze(self.critic(states))

            new_probs = dist.log_prob(actions)
            ratios = torch.exp(new_probs - old_probs)
            advantages = returns - new_values.detach()

            actor_loss = -(ratios * advantages).mean()
            critic_loss = torch.nn.functional.mse_loss(returns, new_values).mean()
            loss = actor_loss + critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.state_memory.clear()
        self.action_memory.clear()
        self.prob_memory.clear()
        self.reward_memory.clear()
        self.terminated_memory.clear()

    def save_model(self):
        path = "../models/{}".format(RUN_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), path + "/actor.pth")


def train():
    env = gym.make("LunarLander-v2")
    writer = SummaryWriter("../summaries/" + RUN_NAME)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, NUM_EPOCHS, GAMMA, LEARNING_RATE)

    score = 0
    scores = []
    for update in range(NUM_UPDATES):
        observation, info = env.reset()
        for update_step in range(BUFFER_SIZE):
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.remember(reward, terminated)
            if terminated:
                scores.append(score)
                writer.add_scalar("Score", score, update * BUFFER_SIZE + update_step)
                score = 0
                observation, info = env.reset()
        agent.learn(observation)
        if (update + 1) % LOG_FREQUENCY == 0:
            print("Update: {}\tAvg. score: {}".format(update, np.mean(scores)))
            scores.clear()

    env.close()
    writer.close()
    agent.save_model()


if __name__ == '__main__':
    train()

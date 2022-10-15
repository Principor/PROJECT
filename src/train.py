import os

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from critic import Critic

NUM_EPISODES = 5000
MAX_STEPS = 500
GAMMA = 0.99
LEARNING_RATE = 0.01
LOG_FREQUENCY = 100
RUN_NAME = "actor_critic"


class Agent:
    def __init__(self, state_size, action_size, gamma, lr):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.action_memory = []
        self.reward_memory = []
        self.value_memory = []

    def choose_action(self, state):
        state_tensor = torch.tensor(state)
        dist = torch.distributions.Categorical(self.actor(state_tensor))
        action = dist.sample()
        self.action_memory.append(dist.log_prob(action))
        self.value_memory.append(self.critic(state_tensor))
        return action.detach().numpy()

    def remember(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        t = np.arange(len(self.reward_memory))
        discounts = self.gamma ** t
        returns = np.array(self.reward_memory) * discounts
        returns = returns[::-1].cumsum()[::-1] / discounts
        returns = torch.tensor(returns, dtype=torch.float32)
        actions = torch.stack(self.action_memory)
        values = torch.squeeze(torch.stack(self.value_memory))
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        actor_loss = -(actions * advantages).mean()
        critic_loss = torch.nn.functional.mse_loss(returns, values).mean()
        loss = actor_loss + critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.action_memory.clear()
        self.reward_memory.clear()
        self.value_memory.clear()

    def save_model(self):
        path = "../models/{}".format(RUN_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), path + "/actor.pth")


def train():
    env = gym.make("LunarLander-v2")
    writer = SummaryWriter("../summaries/" + RUN_NAME)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, GAMMA, LEARNING_RATE)

    scores = []
    for episode in range(NUM_EPISODES):
        score = 0
        observation, info = env.reset()
        for step in range(MAX_STEPS):
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.remember(reward)
            if terminated or truncated:
                break
        scores.append(score)
        writer.add_scalar("Score", score, episode)
        agent.learn()
        if (episode + 1) % LOG_FREQUENCY == 0:
            print("Episode: {}\t\tScore: {}".format(episode, np.mean(scores[-LOG_FREQUENCY:])))
    env.close()
    writer.close()
    agent.save_model()


if __name__ == '__main__':
    train()

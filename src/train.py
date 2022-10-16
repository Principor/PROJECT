import os

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from critic import Critic

NUM_UPDATES = 200
UPDATE_STEPS = 5000
MAX_EPISODE_STEPS = 500
GAMMA = 0.99
LEARNING_RATE = 0.01
LOG_FREQUENCY = 10
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
        self.terminated_memory = []
        self.value_memory = []

    def choose_action(self, state):
        state_tensor = torch.tensor(state)
        dist = torch.distributions.Categorical(self.actor(state_tensor))
        action = dist.sample()
        self.action_memory.append(dist.log_prob(action))
        self.value_memory.append(self.critic(state_tensor))
        return action.detach().numpy()

    def remember(self, reward, terminated):
        self.reward_memory.append(reward)
        self.terminated_memory.append(terminated)

    def learn(self, next_state):
        current_return = self.critic(torch.tensor(next_state))
        returns = []
        for reward, terminated in reversed(list(zip(self.reward_memory, self.terminated_memory))):
            current_return = reward + self.gamma * current_return * (1 - terminated)
            returns.insert(0, current_return)
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
        self.terminated_memory.clear()
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

    observation, info = env.reset()
    score = episode_step = 0
    episodes = 0
    scores = []
    for update in range(NUM_UPDATES):
        for _ in range(UPDATE_STEPS):
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.remember(reward, terminated)
            if terminated or truncated or episode_step == MAX_EPISODE_STEPS:
                scores.append(score)
                episodes += 1
                writer.add_scalar("Score", score, episodes)
                score = episode_step = 0
                observation, info = env.reset()
            episode_step += 1
        agent.learn(observation)
        if (update + 1) % LOG_FREQUENCY == 0:
            print("Update: {}\tAvg. score: {}".format(update, np.mean(scores)))
            scores.clear()

    env.close()
    writer.close()
    agent.save_model()


if __name__ == '__main__':
    train()

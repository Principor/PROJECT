import gym
import torch
import numpy as np
from matplotlib import pyplot as plt

from actor import Actor

NUM_EPISODES = 2000
GAMMA = 0.99
LOG_FREQUENCY = 100
ROLLING_AVG = 10


class Agent:
    def __init__(self, state_size, action_size, gamma):
        self.actor = Actor(state_size, action_size)
        self.gamma = gamma
        self.action_memory = []
        self.reward_memory = []

    def choose_action(self, state):
        state_tensor = torch.tensor(state)
        dist = torch.distributions.Categorical(self.actor(state_tensor))
        action = dist.sample()
        self.action_memory.append(dist.log_prob(action))
        return action.detach().numpy()

    def remember(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        t = np.arange(len(self.reward_memory))
        discounts = GAMMA ** t
        returns = np.array(self.reward_memory) * discounts
        returns = returns[::-1].cumsum()[::-1] / discounts

        self.action_memory.clear()
        self.reward_memory.clear()


def train():
    env = gym.make("LunarLander-v2")
    agent = Agent(env.observation_space.shape[0], env.action_space.n, GAMMA)
    scores = []
    for episode in range(NUM_EPISODES):
        terminated = False
        score = 0
        observation, info = env.reset()
        while not terminated:
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.remember(reward)
        scores.append(score)
        agent.learn()
        if (episode + 1) % LOG_FREQUENCY == 0:
            print("Episode: {}\t\tScore: {}".format(episode, np.mean(scores[-LOG_FREQUENCY:])))
    env.close()
    plt.plot(np.convolve(np.array(scores), np.ones(ROLLING_AVG), 'valid') / ROLLING_AVG)
    plt.show()


if __name__ == '__main__':
    train()

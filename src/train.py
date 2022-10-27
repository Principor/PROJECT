import os

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from critic import Critic

NUM_UPDATES = 100
BATCH_SIZE = 512
NUM_BATCHES = 10
BUFFER_SIZE = NUM_BATCHES * BATCH_SIZE

NUM_EPOCHS = 20
EPSILON = 0.2
GAMMA = 0.99
LEARNING_RATE = 0.0003

HIDDEN_SIZE = 128

LOG_FREQUENCY = 10
RUN_NAME = "ppo"


def normalise(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def prepare_state(state):
    return torch.tensor(state)


class Agent:
    def __init__(self, state_size, action_size, hidden_size, num_epochs, epsilon, gamma, lr):
        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.eps = epsilon
        self.gamma = gamma

        self.state_memory = []
        self.action_memory = []
        self.prob_memory = []
        self.reward_memory = []
        self.terminated_memory = []

    def choose_action(self, state):
        with torch.no_grad():
            state = prepare_state(state)
            dist = self.actor(state)
            action = dist.sample()
            prob = dist.log_prob(action)

        self.state_memory.append(state)
        self.action_memory.append(action.numpy())
        self.prob_memory.append(prob.numpy())

        return action.detach().numpy()

    def remember(self, reward, terminated):
        self.reward_memory.append(reward)
        self.terminated_memory.append(terminated)

    def calculate_returns(self, final_value):
        current_return = final_value
        returns = []
        for reward, terminated in reversed(list(zip(self.reward_memory, self.terminated_memory))):
            current_return = reward + self.gamma * current_return * (1 - terminated)
            returns.insert(0, current_return)
        returns = normalise(np.array(returns))
        return returns

    def learn(self, next_state):
        all_returns = self.calculate_returns(self.critic(torch.tensor(next_state)).item())
        all_states = np.stack(self.state_memory)
        all_actions = np.stack(self.action_memory)
        all_probs = np.stack(self.prob_memory)

        for _ in range(self.num_epochs):

            indices = np.arange(BUFFER_SIZE)
            np.random.shuffle(indices)
            indices = np.split(indices, NUM_BATCHES)

            batches = [(
                torch.tensor(all_returns[batch_indices], dtype=torch.float32),
                torch.tensor(all_states[batch_indices]),
                torch.tensor(all_actions[batch_indices]),
                torch.tensor(all_probs[batch_indices])
            ) for batch_indices in indices]

            for batch in batches:
                returns, states, actions, old_probs = batch
                dist = self.actor(states)
                new_values = torch.squeeze(self.critic(states))

                new_probs = dist.log_prob(actions)
                ratios = torch.exp(new_probs - old_probs)
                advantages = torch.unsqueeze(returns - new_values.detach(), dim=-1)

                unclipped = ratios * advantages
                clipped = torch.clip(ratios, 1-self.eps, 1+self.eps) * advantages

                actor_loss = -torch.min(unclipped, clipped).mean()
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
    env = gym.make("LunarLanderContinuous-v2")
    writer = SummaryWriter("../summaries/" + RUN_NAME)
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], HIDDEN_SIZE, NUM_EPOCHS, EPSILON, GAMMA,
                  LEARNING_RATE)

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

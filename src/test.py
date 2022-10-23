from actor import Actor
import gym
import torch

NUM_STEPS = 1000

env = gym.make("LunarLander-v2", render_mode="human")
actor = Actor(env.observation_space.shape[0], env.action_space.n)
actor.load_state_dict(torch.load("../models/ppo/actor.pth"))

if __name__ == '__main__':
    observation, info = env.reset()
    for _ in range(NUM_STEPS):
        action = actor(torch.tensor(observation)).sample().item()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

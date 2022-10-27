import gym
import torch

from actor import Actor

NUM_STEPS = 5000
HIDDEN_SIZE = 128

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], HIDDEN_SIZE)
    actor.load_state_dict(torch.load("../models/ppo/actor.pth"))

    observation, info = env.reset()
    for _ in range(NUM_STEPS):
        action = actor(torch.tensor(observation)).sample().detach().numpy()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

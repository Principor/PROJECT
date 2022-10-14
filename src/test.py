from actor import Actor
import gym
import torch

NUM_STEPS = 1000

env = gym.make("LunarLander-v2", render_mode="human")
actor = Actor(env.observation_space.shape[0], env.action_space.n)
actor.load_state_dict(torch.load("../models/policy_gradient/actor.pth"))

if __name__ == '__main__':
    observation, info = env.reset()
    for _ in range(NUM_STEPS):
        action = torch.argmax(actor(torch.tensor(observation))).detach().numpy()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

import gym
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from model import Model

# Parameters
NUM_STEPS = 5000
HIDDEN_SIZE = 128

if __name__ == '__main__':
    """
    Test the trained model
    """

    env = DummyVecEnv([lambda: gym.make("LunarLanderContinuous-v2")])
    env = VecNormalize.load("../models/normaliser", env)    # Load normaliser generated during training so inputs match
    actor = Model(env.observation_space.shape[0], env.action_space.shape[0], HIDDEN_SIZE)
    actor.load_state_dict(torch.load("../models/ppo/model.pth"))

    observation = env.reset()
    for _ in range(NUM_STEPS):
        action = actor(torch.tensor(observation, dtype=torch.float32))[0].sample().detach().numpy()
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        if done:
            observation = env.reset()

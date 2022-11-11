import gym
import numpy as np


class RacecarDrivingEnv(gym.Env):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 60
    }

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.full(4, -np.inf),
            high=np.full(4, np.inf),
        )

    def step(self, action):
        pass

    def reset(self, seed=None, options=None, ):
        return self.observation_space.sample(), {}

    def render(self):
        pass

    def close(self):
        pass

import time

import gym
import numpy as np
import pybullet as p

from src.env.racecar_driving.resources import car


TIME_STEP = 0.01


class RacecarDrivingEnv(gym.Env):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 60
    }

    def __init__(self, gui=False):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.full(4, -np.inf),
            high=np.full(4, np.inf),
        )

        self.gui = gui

        p.connect(p.GUI if self.gui else p.DIRECT)
        p.setTimeStep(TIME_STEP)
        p.setGravity(0, 0, -10)

        plane_collision_shape = p.createCollisionShape(p.GEOM_PLANE)
        plane_visual_shape = p.createVisualShape(p.GEOM_PLANE)
        ground = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=plane_collision_shape,
                                   baseVisualShapeIndex=plane_visual_shape)
        p.changeDynamics(ground, -1,
                         restitution=0.9)

        self.car = None

    def step(self, action):
        p.stepSimulation()
        self.car.update(-1, 0, TIME_STEP)
        return self.observation_space.sample(), 0, False, False, {}

    def reset(self, seed=None, options=None, ):
        if self.car is not None:
            self.car.remove()
        self.car = car.Car()
        return self.observation_space.sample(), {}

    def render(self):
        time.sleep(TIME_STEP)

    def close(self):
        p.disconnect()

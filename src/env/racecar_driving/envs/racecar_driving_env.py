import math
import time

import gym
import numpy as np
import pybullet as p

from src.env.racecar_driving.resources import car
from src.env.racecar_driving.resources import util


TIME_STEP = 0.01


def get_distance(position1, position2):
    x1, y1 = position1
    x2, y2 = position2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


class RacecarDrivingEnv(gym.Env):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 60
    }

    def __init__(self, gui=False):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1], dtype=np.float64),
            high=np.array([1, 1], dtype=np.float64)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.full(8, -np.inf),
            high=np.full(8, np.inf),
        )

        self.gui = gui

        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setTimeStep(TIME_STEP, physicsClientId=self.client)
        p.setGravity(0, 0, -10, physicsClientId=self.client)

        plane_collision_shape = p.createCollisionShape(p.GEOM_PLANE)
        plane_visual_shape = p.createVisualShape(p.GEOM_PLANE)
        ground = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=plane_collision_shape,
                                   baseVisualShapeIndex=plane_visual_shape,
                                   physicsClientId=self.client)
        p.changeDynamics(ground, -1,
                         restitution=0.9)

        self.car = None

        goal_shape = p.createVisualShape(p.GEOM_SPHERE, radius=1, rgbaColor=[0, 1, 0, 1])
        self.goal_body = p.createMultiBody(baseMass=0,
                                           basePosition=(0, 0, 1),
                                           baseVisualShapeIndex=goal_shape,
                                           physicsClientId=self.client)
        self.goal_position = (0, 0)

        self.previous_position = self.velocity = (0, 0)

    def step(self, action):
        p.stepSimulation(physicsClientId=self.client)
        self.car.update(-1, 0, TIME_STEP)
        new_position = self.get_car_position()
        previous_distance = get_distance(self.previous_position, self.goal_position)
        new_distance = get_distance(new_position, self.goal_position)
        self.velocity = (np.array(new_position) - self.previous_position) / TIME_STEP
        self.previous_position = new_position
        return self.get_observation(), previous_distance-new_distance, False, {}

    def reset(self, seed=None, options=None):
        if self.car is not None:
            self.car.remove()
        self.car = car.Car(self.client)
        self.move_goal()
        self.previous_position = self.velocity = (0, 0)
        return self.get_observation()

    def render(self, mode="human"):
        time.sleep(TIME_STEP)

    def close(self):
        p.disconnect(physicsClientId=self.client)

    def move_goal(self):
        valid_position = False
        while not valid_position:
            x, y = self.goal_position = (np.random.random(2) - 0.5) * 20
            p.resetBasePositionAndOrientation(self.goal_body, posObj=(x, y, 1), ornObj=(0, 0, 0, 1))
            valid_position = get_distance((x, y), self.get_car_position()) > 3

    def get_car_position(self):
        (x, y, _), _ = self.car.get_transform()
        return x, y

    def get_observation(self):
        car_x, car_y = self.get_car_position()
        velocity_x, velocity_y = self.velocity
        goal_x, goal_y = self.goal_position
        dir_x, dir_y, _ = util.transform_direction(self.car.get_transform(), util.make_vector(0, 1, 0))
        return np.array([
            car_x, car_y,
            velocity_x, velocity_y,
            goal_x, goal_y,
            dir_x, dir_y,
        ], dtype=np.float32)



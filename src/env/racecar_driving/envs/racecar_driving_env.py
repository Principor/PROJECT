import math
import random
import time

import gym
import numpy as np
import pybullet as p

from src.env.racecar_driving.resources import car
from src.env.racecar_driving.resources.util import Vector2

TIME_STEP = 0.01


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
            low=np.full(6, -np.inf),
            high=np.full(6, np.inf),
        )

        self.gui = gui

        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setTimeStep(TIME_STEP, physicsClientId=self.client)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(cameraDistance=40,
                                     cameraYaw=0,
                                     cameraPitch=-45,
                                     cameraTargetPosition=(0, 0, 0))

        plane_collision_shape = p.createCollisionShape(p.GEOM_PLANE)
        plane_visual_shape = p.createVisualShape(p.GEOM_PLANE)
        ground = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=plane_collision_shape,
                                   baseVisualShapeIndex=plane_visual_shape,
                                   physicsClientId=self.client)
        p.changeDynamics(ground, -1,
                         restitution=0.9)

        self.car = None

        waypoint_shape = p.createVisualShape(p.GEOM_SPHERE, radius=1, rgbaColor=[0, 1, 0, 1])
        self.waypoint_body = p.createMultiBody(baseMass=0,
                                               basePosition=(0, 0, -1),
                                               baseVisualShapeIndex=waypoint_shape,
                                               physicsClientId=self.client)

        self.previous_position = Vector2(0, 0)
        self.velocity = Vector2(0, 0)

        self.checkpoints = [
            Vector2(-30, -10), Vector2(-30, 10), Vector2(-20, 20), Vector2(20, 20), Vector2(30, 10), Vector2(20, 0),
            Vector2(10, 10), Vector2(0, 10), Vector2(-10, 0), Vector2(-10, -10), Vector2(-20, -20)
        ]
        for i in range(len(self.checkpoints)):
            p.addUserDebugLine(self._get_checkpoint(i).make_3d(0.1).tuple(),
                               self._get_checkpoint(i + 1).make_3d(0.1).tuple(),
                               lineColorRGB=(1, 0, 0),
                               lineWidth=1,
                               physicsClientId=self.client)
        self.checkpoint_index = 0

        self.steps = 0

    def step(self, action):
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)
            self.car.update(action[0], action[1], TIME_STEP)
            if self.gui:
                time.sleep(TIME_STEP)

        current_position = self._get_car_position()
        while (current_distance := current_position.get_distance(self._get_goal_position())) < 3:
            self.checkpoint_index = (self.checkpoint_index + 1) % len(self.checkpoints)
            self._move_waypoint()
        previous_distance = self.previous_position.get_distance(self._get_goal_position())
        reward = previous_distance - current_distance

        self.velocity = (current_position - self.previous_position) / TIME_STEP
        self.previous_position = current_position
        self.steps += 1

        return self._get_observation(), reward, self.steps >= 200, {}

    def reset(self, seed=None, options=None):
        if self.car is not None:
            self.car.remove()

        self.checkpoint_index = random.randrange(len(self.checkpoints))
        self._move_waypoint()

        start_position = self._get_checkpoint(self.checkpoint_index-1)
        difference = (self._get_goal_position() - start_position).tuple()
        direction = math.atan2(difference[1], difference[0]) - math.pi / 2

        print(start_position, self._get_goal_position(), difference)

        self.car = car.Car(self.client, start_position.make_3d(1.5).tuple(), p.getQuaternionFromEuler((0, 0, direction)))
        self.previous_position = Vector2(0, 0)
        self.velocity = Vector2(0, 0)
        self.steps = 0
        return self._get_observation()

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client)

    def _move_waypoint(self):
        position = self._get_goal_position().make_3d().tuple()
        p.resetBasePositionAndOrientation(self.waypoint_body, posObj=position, ornObj=(0, 0, 0, 1))

    def _get_car_position(self):
        return self.car.get_transform().position.get_xy()

    def _get_goal_position(self):
        return self._get_checkpoint(self.checkpoint_index)

    def _get_checkpoint(self, index):
        return self.checkpoints[index % len(self.checkpoints)]

    def _get_observation(self):
        car_position = self.car.get_transform().position
        points = [
            self.velocity.make_3d(),
            self._get_goal_position().make_3d() - car_position,
            self._get_checkpoint(self.checkpoint_index + 1).make_3d() - car_position,
        ]
        observation = []
        car_inverse = self.car.get_transform().invert()
        for point in points:
            local = car_inverse.transform_direction(point).tuple()
            observation.append(local[0])
            observation.append(local[1])

        return np.array(observation, dtype=np.float32)

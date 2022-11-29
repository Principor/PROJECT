import math
import random
import time

import gym
import numpy as np
import pybullet as p

from src.env.racecar_driving.resources import car
from src.env.racecar_driving.resources.util import Vector2
from src.env.racecar_driving.resources.bezier import Bezier

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
            low=np.full(16, -np.inf),
            high=np.full(16, np.inf),
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

        self.bezier = Bezier(
            Vector2(-114.67, 73.08), Vector2(-131.89, 89.42), Vector2(-103.53, 115.41), Vector2(-86.17, 100.00),
            Vector2(-73.36, 88.62), Vector2(-60.55, 77.24), Vector2(-47.74, 65.87), Vector2(-28.08, 48.42),
            Vector2(-15.36, 85.73), Vector2(3.95, 75.28), Vector2(24.56, 64.11), Vector2(59.82, 7.75),
            Vector2(74.23, -3.90), Vector2(89.52, -16.26), Vector2(119.91, -8.10), Vector2(128.87, -25.71),
            Vector2(135.91, -39.54), Vector2(141.08, -87.25), Vector2(117.49, -87.50), Vector2(46.57, -88.24),
            Vector2(-24.34, -88.99),  Vector2(-95.26, -89.73), Vector2(-134.60, -90.14), Vector2(-99.41, -28.48),
            Vector2(-66.72, -46.84), Vector2(-13.31, -76.84), Vector2(13.68, -48.74), Vector2(-1.46, -34.37),
            Vector2(-39.20, 1.45), Vector2(-76.93, 37.26)
        )
        self.bezier.draw_lines(self.client)

        self.previous_progress = 0
        self.segment_index = 0

        self.steps = 0

    def step(self, action):
        for _ in range(SIM_STEPS):
            p.stepSimulation(physicsClientId=self.client)
            self.car.update(action[0], action[1], TIME_STEP)
            if self.gui:
                time.sleep(TIME_STEP)
        current_position = self._get_car_position()

        correct_segment = False
        t = 0
        while not correct_segment:
            t, _ = self.bezier.get_distance_from_curve(current_position, self.segment_index)
            if t == 1:
                self.segment_index += 1
                self._move_waypoint()
            else:
                break

        current_progress = self.bezier.get_total_progress(self.segment_index, t)
        progress_difference = current_progress - self.previous_progress
        total_length = self.bezier.get_total_length()
        if progress_difference > total_length / 2:
            progress_difference -= total_length
        elif progress_difference < -total_length / 2:
            progress_difference += total_length

        reward = progress_difference

        self.velocity = (current_position - self.previous_position) / TIME_STEP
        self.previous_position = current_position
        self.previous_progress = current_progress
        self.steps += 1

        return self._get_observation(), reward, self.steps >= 200, {}

    def reset(self, seed=None, options=None):
        if self.car is not None:
            self.car.remove()

        self.segment_index = random.randrange(self.bezier.num_segments)
        self._move_waypoint()

        start_position = self._get_checkpoint(self.segment_index)
        difference = (self._get_goal_position() - start_position).tuple()
        direction = math.atan2(difference[1], difference[0]) - math.pi / 2

        self.car = car.Car(self.client, start_position.make_3d(1.5).tuple(),
                           p.getQuaternionFromEuler((0, 0, direction)))
        self.previous_progress = 0
        self.previous_position = Vector2(0, 0)
        self.velocity = Vector2(0, 0)
        self.steps = 0
        return self._get_observation()

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client)

    def _move_waypoint(self):
        position = self._get_goal_position().make_3d(1).tuple()
        p.resetBasePositionAndOrientation(self.waypoint_body, posObj=position, ornObj=(0, 0, 0, 1))

    def _get_car_position(self):
        return self.car.get_transform().position.get_xy()

    def _get_goal_position(self):
        return self._get_checkpoint(self.segment_index+1)

    def _get_checkpoint(self, index):
        return self.bezier.get_segment_point(index, 0)

    def _get_observation(self):
        car_position = self.car.get_transform().position
        points = [self.velocity.make_3d()]
        points += [self.bezier.get_segment_point(self.segment_index, i).make_3d() - car_position for i in range(7)]
        observation = []
        car_inverse = self.car.get_transform().invert()
        for point in points:
            local = car_inverse.transform_direction(point).tuple()
            observation.append(local[0])
            observation.append(local[1])
        return np.array(observation, dtype=np.float32)

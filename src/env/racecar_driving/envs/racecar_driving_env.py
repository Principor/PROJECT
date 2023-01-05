import math
import os
import pickle
import random
import time

import gym
import numpy as np
import pybullet as p

from src.env.racecar_driving.resources import car
from src.env.racecar_driving.resources.util import Vector2
from src.env.racecar_driving.resources.bezier import Bezier

# Simulation Parameters
TIME_STEP = 0.01
SIM_STEPS = 10
TRACK_WIDTH = 20


class RacecarDrivingEnv(gym.Env):
    """
    Gym environment for driving a car around a race track

    :param gui: Create environment with debugger window
    :param random_start: Whether to start from a random point or fixed point
    :param save_telemetry: Record positioning and inputs of the car at all points
    """

    metadata = {
        'render_modes': ['human'],
    }

    def __init__(self, gui=False, random_start=True, save_telemetry=False):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1], dtype=np.float64),
            high=np.array([1, 1], dtype=np.float64)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.full(16, -np.inf),
            high=np.full(16, np.inf),
        )

        self.gui = gui
        self.random_start = random_start
        self.save_telemetry = save_telemetry

        self.client = p.connect(p.GUI if self.gui else p.DIRECT, options='--width=800 --height=800')
        p.resetDebugVisualizerCamera(cameraDistance=30,
                                     cameraYaw=0,
                                     cameraPitch=-45,
                                     cameraTargetPosition=(0, -50, 0))
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        p.setTimeStep(TIME_STEP, physicsClientId=self.client)
        p.setGravity(0, 0, -10, physicsClientId=self.client)

        plane_collision_shape = p.createCollisionShape(p.GEOM_PLANE)
        plane_visual_shape = p.createVisualShape(p.GEOM_PLANE)
        ground = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=plane_collision_shape,
                                   baseVisualShapeIndex=plane_visual_shape,
                                   physicsClientId=self.client)
        p.changeDynamics(ground, -1, restitution=0.9)

        self.car = None

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
        self.bezier.draw_lines(self.client, TRACK_WIDTH)

        self.previous_progress = 0
        self.segment_index = 0

        self.steps = 0

        self.telemetry = []

    def step(self, action):
        """
        Perform one time-step of the environment

        :param action: throttle/brake and steering angle
        :return: observation, reward, terminal, info
        """
        for _ in range(SIM_STEPS):
            p.stepSimulation(physicsClientId=self.client)
            self.car.update(action[0], action[1], TIME_STEP)
            if self.gui:
                time.sleep(TIME_STEP)
        current_position = self._get_car_position()

        # Increase segment index if car moved into next segment
        correct_segment = False
        t, distance = None, None
        while not correct_segment:
            t, distance = self.bezier.get_distance_from_curve(current_position, self.segment_index)
            if t == 1:
                self.segment_index += 1
            else:
                break

        # Calculate distance along track travelled this step
        current_progress = self.bezier.get_total_progress(self.segment_index, t)
        progress_difference = current_progress - self.previous_progress
        total_length = self.bezier.get_total_length()
        if progress_difference > total_length / 2:
            progress_difference -= total_length
        elif progress_difference < -total_length / 2:
            progress_difference += total_length

        reward = progress_difference

        self.velocity = (current_position - self.previous_position) / (TIME_STEP * SIM_STEPS)
        self.previous_position = current_position
        self.previous_progress = current_progress
        self.steps += 1

        if self.save_telemetry:
            self.telemetry.append([self.segment_index, t, current_position, self.velocity.magnitude(),
                                   action[0], action[1]])

        return self._get_observation(), reward, self.steps >= 1000 or distance > TRACK_WIDTH / 2, {}

    def reset(self, seed=None, options=None):
        """
        Start a new episode, car will be sent to random point on the track

        :param seed: None
        :param options:  None
        :return: observation
        """
        self._output_telemetry()
        self.telemetry = []

        if self.car is not None:
            self.car.remove()

        if self.random_start:
            self.segment_index = random.randrange(self.bezier.num_segments)
            t = random.random()
        else:
            self.segment_index = 6
            t = 0

        start_position = self.bezier.get_curve_point(self.segment_index, t)
        direction = self.bezier.get_direction(self.segment_index, t).tuple()
        angle = math.atan2(direction[1], direction[0]) - math.pi / 2

        self.car = car.Car(self.client, start_position.make_3d(1.5).tuple(),
                           p.getQuaternionFromEuler((0, 0, angle)))
        self.previous_progress = self.bezier.get_total_progress(self.segment_index, t)
        self.previous_position = Vector2(0, 0)
        self.velocity = Vector2(0, 0)
        self.steps = 0
        return self._get_observation()

    def render(self, mode="human"):
        pass

    def close(self):
        """
        Close the environment
        """
        self._output_telemetry()
        p.disconnect(physicsClientId=self.client)

    def _get_car_position(self):
        # Get current position of the car in 2D
        return self.car.get_transform().position.get_xy()

    def _get_observation(self):
        # Get the current observation: velocity of the car and positions of current and next segment control points,
        # all in local space
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

    def _output_telemetry(self):
        if len(self.telemetry) == 0:
            return
        if not os.path.exists("../telemetry"):
            os.makedirs("../telemetry")
        with open("../telemetry/output.pkl", 'wb') as file:
            pickle.dump({"track": self.bezier, "track_width": TRACK_WIDTH, "telemetry": self.telemetry}, file)

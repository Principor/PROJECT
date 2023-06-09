import math
import os
import pickle
import random
import time

import gym
import numpy as np
import pybullet as p
from Box2D import b2World, b2EdgeShape, b2RayCastCallback, b2Vec2

from racecar_driving.resources.car_generator import CarGenerator
from src.env.racecar_driving.resources.util import Vector2
from src.env.racecar_driving.resources.bezier import Bezier

# Simulation Parameters
TIME_STEP = 0.01
SIM_STEPS = 10
TRACK_WIDTH = 20
MAX_RAY_LENGTH = 100


class RacecarDrivingEnv(gym.Env):
    """
    Gym environment for driving a car around a race track

    :param gui: Create environment with debugger window
    :param random_start: Whether to start from a random point or fixed point
    :param save_telemetry: Record positioning and inputs of the car at all points
    :param car_index: The index of the car to use
    :param track_list: List of names of tracks to use
    :param transform_tracks: Whether tracks should be transformed
    """

    metadata = {
        'render_modes': ['human'],
    }

    def __init__(self, gui=False, random_start=True, save_telemetry=False, car_index=-1, track_list=None,
                 transform_tracks=True):
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

        self.car_generator = CarGenerator(self.client, car_index)
        self.car = None

        self.previous_position = Vector2(0, 0)
        self.velocity = Vector2(0, 0)

        self.track_list = track_list if track_list else Bezier.list_saves()[:-1]
        self.transform_tracks = transform_tracks

        self.bezier = None
        self.debug_lines = []
        self.box2d_world = None

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

        # Generate new track
        self.bezier = Bezier.load(random.choice(self.track_list))
        if self.transform_tracks:
            if random.choice([True, False]):
                self.bezier.mirror()
            if random.choice([True, False]):
                self.bezier.reverse()
            self.bezier.add_noise()

        # Set up track edges in debug view and 2D physics world
        track_lines = self.bezier.get_lines(TRACK_WIDTH)
        self._draw_lines(track_lines)
        self.box2d_world = b2World(gravity=(0, -10))
        for start, end in track_lines:
            self.box2d_world.CreateStaticBody(
                shapes=b2EdgeShape(vertices=[start.get_xy().tuple(), end.get_xy().tuple()]),
                position=(1, 0)
            )

        self._output_telemetry()
        self.telemetry = []

        # Spawn car
        if self.random_start:
            self.segment_index = random.randrange(self.bezier.num_segments)
            t = random.random()
        else:
            self.segment_index = 6
            t = 0
        start_position = self.bezier.get_curve_point(self.segment_index, t)
        direction = self.bezier.get_direction(self.segment_index, t).tuple()
        angle = math.atan2(direction[1], direction[0]) - math.pi / 2
        self.car = self.car_generator.reset_car(
            start_position.make_3d(1.5).tuple(),
            p.getQuaternionFromEuler((0, 0, angle))
        )

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

    def _draw_lines(self, track_lines):
        """
        Draw the track limits in the pybullet renderer

        :param track_lines: The width of the track
        """
        if not self.gui:
            return
        for line in self.debug_lines:
            p.removeUserDebugItem(line, physicsClientId=self.client)
        for start, end in track_lines:
            self.debug_lines.append(p.addUserDebugLine(
                start.tuple(),
                end.tuple(),
                lineColorRGB=(1, 0, 0),
                lineWidth=2,
                physicsClientId=self.client
            ))

    def _get_car_position(self):
        # Get current position of the car in 2D
        return self.car.get_transform().position.get_xy()

    def _get_observation(self):
        # Get state/observation for current time-step

        observation = []
        car_transform = self.car.get_transform()

        # Velocity
        inverse_transform = self.car.get_transform().invert()
        relative_velocity = inverse_transform.transform_direction(self.velocity.make_3d())
        vel_x, vel_y = relative_velocity.get_xy().tuple()
        observation += [vel_x, vel_y]

        # LIDAR
        side_angles = np.array([2.5, 5, 10, 15, 30, 60])
        all_angles = np.concatenate((-side_angles[::-1], [0], side_angles))
        all_angles = [math.radians(angle) for angle in all_angles]
        forward = car_transform.transform_direction(Vector2(0, 1).make_3d())
        ray_start = b2Vec2(*car_transform.position.get_xy().tuple())
        x, y = forward.get_xy().tuple()
        for theta in all_angles:
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            dir_x, dir_y = (x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta)
            ray_end = ray_start + b2Vec2(dir_x, dir_y) * MAX_RAY_LENGTH
            callback = RayCastClosestCallback()
            self.box2d_world.RayCast(callback, ray_start, ray_end)
            observation.append(callback.fraction * MAX_RAY_LENGTH)

        # Car Configuration
        observation += self.car.get_configuration()

        return np.array(observation, dtype=np.float32)

    def _output_telemetry(self):
        # Save telemetry to a file
        if len(self.telemetry) == 0:
            return
        if not os.path.exists("../telemetry"):
            os.makedirs("../telemetry")
        with open("../telemetry/output.pkl", 'wb') as file:
            pickle.dump({"track": self.bezier, "track_width": TRACK_WIDTH, "telemetry": self.telemetry}, file)


class RayCastClosestCallback(b2RayCastCallback):
    """
    Callback to find distance of closest hit
    Copied from raycast.py at https://github.com/pybox2d/pybox2d
    """

    def __repr__(self):
        return 'Closest hit'

    def __init__(self):
        b2RayCastCallback.__init__(self)
        self.fixture = None
        self.hit = False
        self.point = None
        self.normal = None
        self.fraction = 1

    def ReportFixture(self, fixture, point, normal, fraction):
        '''
        Called for each fixture found in the query. You control how the ray
        proceeds by returning a float that indicates the fractional length of
        the ray. By returning 0, you set the ray length to zero. By returning
        the current fraction, you proceed to find the closest point. By
        returning 1, you continue with the original ray clipping. By returning
        -1, you will filter out the current fixture (the ray will not hit it).
        '''
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        self.fraction = fraction

        return fraction


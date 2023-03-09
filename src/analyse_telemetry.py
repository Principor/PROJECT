import pickle

import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt

from model import Model, state_to_tensor
import racecar_driving

RUN_NAME = "ff"
new_telemetry = True
plot_throttle = False
plot_position = False
plot_speed = True

def main():
    if new_telemetry:
        env = DummyVecEnv([lambda: gym.make('RacecarDriving-v0', save_telemetry=True, random_start=False)])
        # Load normaliser generated during training so inputs match
        env = VecNormalize.load("../models/{}/normaliser".format(RUN_NAME), env)
        actor = Model.load_model("../models/{}/model.pth".format(RUN_NAME))

        done = False
        observation = env.reset()
        while not done:
            action = actor(state_to_tensor(observation))[0].sample().detach().numpy().squeeze(0)
            observation, reward, done, info = env.step(action)

    with open('../telemetry/output.pkl', 'rb') as file:
        data = pickle.load(file)
        track = data["track"]
        track_width = data["track_width"]
        telemetry = data["telemetry"]

        start_index = 1
        while telemetry[start_index][0] == telemetry[0][0]:
            start_index += 1

        end_index = len(telemetry) - 1
        while telemetry[end_index][0] > track.num_segments * 2 - 1:
            end_index -= 1

        telemetry = telemetry[start_index:end_index]

        progress = np.array([track.get_total_progress(item[0], item[1]) for item in telemetry])
        progress -= np.min(progress)

        if plot_throttle:
            throttle = np.clip(np.array(telemetry)[:, 4], 0, 1)
            brake = np.clip(-np.array(telemetry)[:, 5], 0, 1)
            plt.plot(progress, throttle, color='g')
            plt.plot(progress, brake, color='r')
            plt.show()

        if plot_position:
            left_points = []
            right_points = []
            car_points = []
            for item in telemetry:
                segment_index = item[0]
                t = item[1]
                car_position = item[2]
                mid_point = track.get_curve_point(segment_index, t)
                right = track.get_direction(segment_index, t).rotate_90()
                left_points.append((mid_point - right * track_width / 2).tuple())
                right_points.append((mid_point + right * track_width / 2).tuple())
                car_points.append(car_position.tuple())

            plt.plot([x for x, _ in left_points], [y for _, y in left_points], color="black")
            plt.plot([x for x, _ in right_points], [y for _, y in right_points], color="black")
            plt.plot([x for x, _ in car_points], [y for _, y in car_points], color="blue")
            plt.show()

        if plot_speed:
            plt.plot(progress, np.array(telemetry)[:, 3])
            plt.show()


if __name__ == '__main__':
    main()

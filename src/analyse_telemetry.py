import argparse
import pickle

import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt

from model import Model, state_to_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="lstm_asymmetric", help="the name of the run")
    parser.add_argument("--car_index", type=int, default=-1, help="the index of the car to use")
    parser.add_argument("--track_name", type=str, default="test", help="the name of the track to use")
    parser.add_argument("--new_telemetry", type=bool, default=True, help="whether new telemetry should be generated")
    parser.add_argument("--plot_throttle", type=bool, default=False,
                        help="whether the throttle/brake should be plotted")
    parser.add_argument("--plot_position", type=bool, default=False, help="whether the position should be plotted")
    parser.add_argument("--plot_speed", type=bool, default=False, help="whether the speed should be plotted")
    return parser.parse_args()


def analyse_telemetry():
    """
    Generate and display telemetry
    """

    args = parse_args()

    # Generate new telemetry
    if args.new_telemetry:
        env = DummyVecEnv([lambda: gym.make('RacecarDriving-v0', save_telemetry=True, random_start=False,
                                            car_index=args.car_index, track_list=[args.track_name],
                                            transform_tracks=False)])
        # Load normaliser generated during training so inputs match
        env = VecNormalize.load("../models/{}/normaliser".format(args.run_name), env)
        env.training = False
        actor = Model.load_model("../models/{}/model.pth".format(args.run_name))

        done = False
        observation = env.reset()
        while not done:
            action = actor(state_to_tensor(observation))[0].sample().detach().numpy().squeeze(0)
            observation, reward, done, info = env.step(action)

    # Analyse the telemetry
    with open('../telemetry/output.pkl', 'rb') as file:
        data = pickle.load(file)
        track = data["track"]
        track_width = data["track_width"]
        telemetry = data["telemetry"]

        if telemetry[0][0] + track.num_segments > telemetry[-1][0]:
            print("Invalid telemetry - 1 lap was not completed!")
            return

        start_index = 1
        while telemetry[start_index][0] == telemetry[0][0]:
            start_index += 1

        end_index = len(telemetry) - 1
        while telemetry[end_index][0] > telemetry[0][0] + track.num_segments:
            end_index -= 1

        telemetry = telemetry[start_index:end_index]

        progress = np.array([track.get_total_progress(item[0], item[1]) for item in telemetry])
        progress -= np.min(progress)

        if args.plot_throttle:
            throttle = np.clip(np.array(telemetry)[:, 4], 0, 1)
            brake = np.clip(-np.array(telemetry)[:, 5], 0, 1)
            plt.plot(progress, throttle, color='g')
            plt.plot(progress, brake, color='r')
            plt.show()

        if args.plot_position:
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

            left_points.append(left_points[0])
            right_points.append(right_points[0])

            plt.plot([x for x, _ in left_points], [y for _, y in left_points], color="black")
            plt.plot([x for x, _ in right_points], [y for _, y in right_points], color="black")
            plt.plot([x for x, _ in car_points], [y for _, y in car_points], color="blue")
            plt.show()

        if args.plot_speed:
            plt.plot(progress, np.array(telemetry)[:, 3])
            plt.show()


if __name__ == '__main__':
    analyse_telemetry()

import argparse

import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from model import Model, state_to_tensor
import racecar_driving


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=bool, default=False, help="create environment with debugger window")
    parser.add_argument("--num_episodes", type=int, default=10, help="number of episodes to perform")
    parser.add_argument("--car_index", type=int, default=0, help="the index of the car to use")
    parser.add_argument("--run_name", type=str, default="lstm_asymmetric", help="the name of the run")
    parser.add_argument("--track_name", type=str, default="test", help="the name of the track to use")
    return parser.parse_args()


def test():
    """
    Test the trained model
    """
    args = parse_args()

    env = DummyVecEnv([lambda: gym.make('RacecarDriving-v0', gui=args.gui, car_index=args.car_index,
                                        track_list=[args.track_name], transform_tracks=False)])
    # Load normaliser generated during training so inputs match
    env = VecNormalize.load("../models/{}/normaliser".format(args.run_name), env)
    env.training = False
    actor = Model.load_model("../models/{}/model.pth".format(args.run_name))

    print()

    scores = []
    observation = env.reset()
    for episode in range(args.num_episodes):
        done = False
        score = 0
        actor.initialise_hidden_states(1)
        while not done:
            action = actor(state_to_tensor(observation))[0].sample().detach().numpy().squeeze(0)
            observation, reward, done, info = env.step(action)
            score += env.get_original_reward().item()
        print("Episode {} score: {}".format(episode, score))
        scores.append(score)
    print("\nMean score: {}".format(sum(scores) / args.num_episodes))


if __name__ == '__main__':
    test()

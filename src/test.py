import gym
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from model import Model, state_to_tensor
import racecar_driving

# Parameters
GUI = True
NUM_EPISODES = 10
CAR_INDEX = 0
RUN_NAME = "ff"

if __name__ == '__main__':
    """
    Test the trained model
    """

    track_list = ["test"]
    env = DummyVecEnv([lambda: gym.make('RacecarDriving-v0', gui=GUI, car_index=CAR_INDEX, track_list=track_list,
                                        transform_tracks=False)])
    # Load normaliser generated during training so inputs match
    env = VecNormalize.load("../models/{}/normaliser".format(RUN_NAME), env)
    env.training = False
    actor = Model.load_model("../models/{}/model.pth".format(RUN_NAME))

    print()

    scores = []
    observation = env.reset()
    for episode in range(NUM_EPISODES):
        done = False
        score = 0
        actor.initialise_hidden_states(1)
        while not done:
            action = actor(state_to_tensor(observation))[0].sample().detach().numpy().squeeze(0)
            observation, reward, done, info = env.step(action)
            score += env.get_original_reward().item()
        print("Episode {} score: {}".format(episode, score))
        scores.append(score)
    print("\nMean score: {}".format(sum(scores) / NUM_EPISODES))

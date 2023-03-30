import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from model.model import Model, state_to_tensor

# Parameters
GUI = False
NUM_EPISODES = 10
CAR_INDEX = 1
RUN_NAME = "ff_unmasked"
TRACK_LIST = ["test"]


def main():
    """
    Test the trained model
    """

    env = DummyVecEnv([lambda: gym.make('RacecarDriving-v0', gui=GUI, car_index=CAR_INDEX, track_list=TRACK_LIST,
                                        transform_tracks=False)])
    # Load normaliser generated during training so inputs match
    env = VecNormalize.load("../models/{}/normaliser".format(RUN_NAME), env)
    env.training = False
    actor = Model.load_model("../models/{}/model.pth".format(RUN_NAME))
    print(actor.state_mask_type)

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


if __name__ == '__main__':
    main()

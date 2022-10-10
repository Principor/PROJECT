import gym

from actor import Actor

NUM_EPISODES = 10

env = gym.make("LunarLander-v2", render_mode="human")
actor = Actor(env.observation_space.shape[0], env.action_space.n)
for episode in range(NUM_EPISODES):
    terminated = False
    observation, info = env.reset()
    while not terminated:
        action = actor(observation)
        observation, reward, terminated, truncated, info = env.step(action)
env.close()

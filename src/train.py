import gym

NUM_EPISODES = 10

env = gym.make("LunarLander-v2", render_mode="human")
for episode in range(NUM_EPISODES):
    terminated = False
    observation, info = env.reset()
    while not terminated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
env.close()

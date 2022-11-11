import gym
import racecar_driving

env = gym.make('RacecarDriving-v0', gui=True)

env.reset()

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()

env.close()

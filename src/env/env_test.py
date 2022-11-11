import gym
import racecar_driving

env = gym.make('RacecarDriving-v0')
env.reset()
env.render()

env.close()

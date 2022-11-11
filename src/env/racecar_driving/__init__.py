from gym.envs.registration import register

register(
    id='RacecarDriving-v0',
    entry_point='racecar_driving.envs:RacecarDrivingEnv'
)

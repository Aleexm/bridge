from gym.envs.registration import register

register(
    id='play-v0',
    entry_point='gym_play.envs:PlayEnv',
)
register(
    id='play-extrahard-v0',
    entry_point='gym_play.envs:PlayExtraHardEnv',
)

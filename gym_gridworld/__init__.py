from gym.envs.registration import register

register(
    id='gridworld-small-v0',
    entry_point='gym_simple_gridworld.envs:GridWorldEnv'
)
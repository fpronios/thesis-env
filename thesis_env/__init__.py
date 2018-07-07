from gym.envs.registration import register

register(
    id='mgrid-v0',
    entry_point='thesis_env.envs:MGridEnv',
)

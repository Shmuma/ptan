__all__ = ('EnvParams', 'register', 'get')

import gym

_current_env_params = None


class EnvParams:
    """
    Simple container for various environment options
    """
    def __init__(self, env):
        assert isinstance(env, gym.Env)
        self.n_actions = env.action_space.n
        self.state_shape = env.observation_space.shape


def register(params):
    global _current_env_params
    _current_env_params = params


def get():
    return _current_env_params

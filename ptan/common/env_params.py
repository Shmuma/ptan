__all__ = ('EnvParams', 'register', 'get')

import gym

_current_env_params = None


class EnvParams:
    """
    Simple container for various environment options
    """
    def __init__(self, n_actions, state_shape):
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.cuda_enabled = False

    def load_runfile(self, run):
        self.cuda_enabled = run.cuda_enabled

    @classmethod
    def from_env(cls, env):
        return EnvParams(n_actions=env.action_space.n, state_shape=env.observation_space.shape)


def register(params):
    global _current_env_params
    _current_env_params = params


def get():
    return _current_env_params

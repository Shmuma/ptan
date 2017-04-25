from unittest import TestCase

import gym
from ptan.common import env_params


class TestEnvParams(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("Acrobot-v1")

    def test_params(self):
        params = env_params.EnvParams.from_env(self.env)
        self.assertEqual(3, params.n_actions)
        self.assertEqual((6,), params.state_shape)

    def test_register(self):
        self.assertIsNone(env_params.get())
        params = env_params.EnvParams.from_env(self.env)
        env_params.register(params)
        self.assertEqual(params, env_params.get())

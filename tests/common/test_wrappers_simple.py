import unittest
import gym
import numpy as np
from ptan.common import wrappers_simple


class SimpleEnv(gym.Env):
    def __init__(self, obs_shape, obs_low, obs_high, obs_dtype=np.float32):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=obs_dtype)
        self._dt = 0

    def reset(self):
        return np.zeros_like(self.observation_space.low)

    def step(self, action):
        delta = -1 if action == 0 else 1
        self._dt += delta
        return np.full_like(self.observation_space.low, self._dt), 1, self._dt == 0, {}


class TestFrameStack1D(unittest.TestCase):
    def test_obs_space(self):
        e = SimpleEnv(obs_shape=(1, ), obs_low=-np.inf, obs_high=np.inf)
        e = wrappers_simple.FrameStack1D(e, 3)
        self.assertIsInstance(e.action_space, gym.spaces.Discrete)

        self.assertIsInstance(e.observation_space, gym.spaces.Box)
        self.assertEquals(e.observation_space.shape, (3, ))
        self.assertEquals(e.observation_space.dtype, np.float32)

        with self.assertRaises(AssertionError):
            e = SimpleEnv(obs_shape=(2,2), obs_low=-np.inf, obs_high=np.inf)
            e = wrappers_simple.FrameStack1D(e, 3)

    def test_reset(self):
        e = SimpleEnv(obs_shape=(2, ), obs_low=-np.inf, obs_high=np.inf)
        e = wrappers_simple.FrameStack1D(e, 3)
        v = e.reset()
        np.testing.assert_array_equal(v, np.zeros((6, )))

    def test_step(self):
        e = SimpleEnv(obs_shape=(2, ), obs_low=-np.inf, obs_high=np.inf)
        e = wrappers_simple.FrameStack1D(e, 3)
        e.reset()
        vv, r, is_done, _ = e.step(0)
        np.testing.assert_array_equal(vv, [0, 0, 0, 0, -1, -1])
        vv, r, is_done, _ = e.step(0)
        np.testing.assert_array_equal(vv, [0, 0, -1, -1, -2, -2])
        vv, r, is_done, _ = e.step(0)
        np.testing.assert_array_equal(vv, [-1, -1, -2, -2, -3, -3])
        vv, r, is_done, _ = e.step(0)
        np.testing.assert_array_equal(vv, [-2, -2, -3, -3, -4, -4])

    def test_done(self):
        e = SimpleEnv(obs_shape=(2, ), obs_low=-np.inf, obs_high=np.inf)
        e = wrappers_simple.FrameStack1D(e, 3)
        e.reset()
        vv, r, is_done, _ = e.step(0)
        np.testing.assert_array_equal(vv, [0, 0, 0, 0, -1, -1])
        vv, r, is_done, _ = e.step(0)
        np.testing.assert_array_equal(vv, [0, 0, -1, -1, -2, -2])
        vv, r, is_done, _ = e.step(1)
        np.testing.assert_array_equal(vv, [-1, -1, -2, -2, -1, -1])
        vv, r, is_done, _ = e.step(1)
        np.testing.assert_array_equal(vv, [-2, -2, -1, -1, 0, 0])
        self.assertTrue(is_done)

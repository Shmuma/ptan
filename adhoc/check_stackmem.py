"""
Script checks functionaly of Lazy frames and stack wrappers.
Should be implemented in a test suite.
"""
import gym
import ptan
import resource

import numpy as np


class GeneratorEnv(gym.Env):
    def __init__(self, dim, cycle, color_planes=1):
        super(GeneratorEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(color_planes, dim, dim), dtype=np.uint8)
        self.value_next = 0
        self.value_cycle = cycle

    def reset(self):
        self.value_next = 0
        return self._next_obs()

    def _next_obs(self):
        res = np.full(self.observation_space.shape, self.value_next, dtype=self.observation_space.dtype)
        self.value_next = (self.value_next + 1) % self.value_cycle
        return res

    def step(self, action):
        return self._next_obs(), 1.0, False, {'next_val': self.value_next}


def check_basic():
    env = GeneratorEnv(dim=10, cycle=2)
    o = env.reset()

    for _ in range(20):
        print(o)
        o = env.step(0)[0]


def check_order():
    env = GeneratorEnv(dim=10, cycle=5)
    env = ptan.common.wrappers.FrameStack(env, 4)

    o = env.reset()
    for _ in range(20):
        o = env.step(0)[0]
    d = np.array(o, copy=False)
    v = d[:, 0, 0]
    if v.tolist() != [2, 3, 4, 0]:
        print("Got wrong result: %s" % v)


def check_memory():
    env = GeneratorEnv(dim=1000, cycle=50)
    env = ptan.common.wrappers.FrameStack(env, 10)
    max_rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    o = env.reset()

    buf = []
    for _ in range(200):
        buf.append(o)
        o = env.step(0)[0]
    delta_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - max_rss_before
    if delta_rss > 300*1024:
        print("Too large RSS used: %.2f MiB" % (delta_rss / 1024))


if __name__ == "__main__":
    check_memory()
    pass


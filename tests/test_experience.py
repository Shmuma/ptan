import itertools
import typing as tt
import numpy as np
import torch

import gymnasium as gym
from ptan import experience, agent


class DummyAgent(agent.BaseAgent):
    def __call__(self, states: agent.States, agent_states: agent.AgentStates = None) -> \
            tt.Tuple[np.ndarray, agent.AgentStates]:
        if isinstance(states, list):
            l = len(states)
        else:
            l = states.shape[0]
        return np.zeros(shape=(l, ), dtype=np.int32), agent_states

    def _net_filter(self, net_out: tt.Any, agent_states: agent.AgentStates) -> \
            tt.Tuple[torch.Tensor, agent.AgentStates]:
        return net_out, agent_states


class CountingEnv(gym.Env):
    def __init__(self, limit: int = 5):
        self.state = 0
        self.limit = limit
        self.observation_space = gym.spaces.Discrete(limit)
        self.action_space = gym.spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, tt.Any] | None = None,
    ) -> tuple[gym.core.ObsType, dict[str, tt.Any]]:
        self.state = 0
        return self.state, {}

    def step(self, action):
        self.state += 1
        done = self.state == self.limit-1
        reward = self.state
        return self.state, reward, done, False, {}


def test_exp_source_single_env_steps(car_env: gym.Env):
    exp_source = experience.ExperienceSource(car_env, DummyAgent(), steps_count=1)
    exp = next(iter(exp_source))
    assert len(exp) == 1
    assert isinstance(exp, tuple)
    assert isinstance(exp[0], experience.Experience)
    assert exp[0].reward == -1
    assert not exp[0].done_trunc

    exp_source = experience.ExperienceSource(car_env, DummyAgent(), steps_count=2)
    exp = next(iter(exp_source))
    assert len(exp) == 2


def test_exp_source_single_env_short_game(cartpole_env: gym.Env):
    exp_source = experience.ExperienceSource(cartpole_env, DummyAgent(), steps_count=1)
    for step_idx, exp in enumerate(exp_source):
        assert len(exp) == 1
        if exp[0].done_trunc:
            assert step_idx < 20
            break


def test_exp_source_many_envs():
    envs = [gym.make("MountainCar-v0") for _ in range(10)]
    exp_source = experience.ExperienceSource(envs, DummyAgent(), steps_count=1)

    exp = next(iter(exp_source))
    assert len(exp) == 1


class StatefulAgent(agent.BaseAgent):
    def __init__(self, action_space):
        super(StatefulAgent, self).__init__()
        self.action_space = action_space

    def initial_state(self):
        return 0

    def __call__(self, states, agent_states):
        new_agent_states = [n+1 for n in agent_states]
        actions = [n % self.action_space.n for n in new_agent_states]
        return np.array(actions, dtype=np.int32), new_agent_states


def test_exp_source_stateful():
    envs = [gym.make("CartPole-v1") for _ in range(2)]

    actions_count = envs[0].action_space.n
    my_agent = StatefulAgent(envs[0].action_space)
    steps = 3
    exp_source = experience.ExperienceSource(envs, my_agent, steps_count=steps)

    for _, exp in zip(range(100), exp_source):
        prev_act: tt.Optional[int] = None
        for e in exp:
            if prev_act is not None:
                assert e.action == (prev_act+1) % actions_count
            prev_act = e.action
        if len(exp) != steps:
            assert exp[-1].done_trunc


#
# class TestExperienceReplayBuffer(TestCase):
#     @classmethod
#     def setUpClass(cls):
#         env = gym.make("MountainCar-v0")
#         cls.source = experience.ExperienceSource(env, agent=DummyAgent())
#
#     def test_len(self):
#         buf = experience.ExperienceReplayBuffer(self.source, buffer_size=2)
#         self.assertEqual(0, len(buf))
#         self.assertEqual([], list(buf))
#
#         buf.populate(1)
#         self.assertEqual(1, len(buf))
#
#         buf.populate(2)
#         self.assertEqual(2, len(buf))
#
#     def test_sample(self):
#         buf = experience.ExperienceReplayBuffer(self.source, buffer_size=10)
#         buf.populate(10)
#         b = buf.sample(4)
#         self.assertEqual(4, len(b))
#
#         buf_ids = list(map(id, buf))
#         check = list(map(lambda v: id(v) in buf_ids, b))
#         self.assertTrue(all(check))
#
#         b = buf.sample(20)
#         self.assertEqual(10, len(b))
#
#
# class TestUtils(TestCase):
#     def test_group_list(self):
#         r = experience._group_list([1, 2, 3], [3])
#         self.assertEqual(r, [[1, 2, 3]])
#         r = experience._group_list([1, 2, 3], [1, 1, 1])
#         self.assertEqual(r, [[1], [2], [3]])
#         r = experience._group_list([1, 2, 3], [1, 2])
#         self.assertEqual(r, [[1], [2, 3]])
#
#
# class TestExperienceSourceFirstLast(TestCase):
#     def test_simple(self):
#         env = CountingEnv()
#         agent = DummyAgent()
#         exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
#         states = []
#         for idx, exp in enumerate(exp_source):
#             states.append(exp.state)
#             if idx > 5:
#                 break
#         # NB: there is no last state(4), as we don't record it
#         self.assertEquals(states, [0, 1, 2, 3, 0, 1, 2])
#

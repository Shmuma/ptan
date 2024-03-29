import collections
import typing as tt
import numpy as np
import pytest
from pytest import approx
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


def test_exp_source_many_envs_counting():
    envs = [CountingEnv(), CountingEnv()]
    exp_source = experience.ExperienceSource(envs, DummyAgent(), steps_count=2)
    data = []
    for _, exp in zip(range(10), exp_source):
        data.append(exp)
    assert data[0] == data[1]
    assert data[2] == data[3]

    e = CountingEnv()
    with pytest.raises(ValueError):
        experience.ExperienceSource([e, e, CountingEnv()], DummyAgent())


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

    rw_steps = exp_source.pop_rewards_steps()
    assert isinstance(rw_steps, list)
    rw = exp_source.pop_total_rewards()
    assert isinstance(rw, list)


def test_firstlast():
    env = CountingEnv()
    agent = DummyAgent()
    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
    states = []
    for idx, exp in enumerate(exp_source):
        states.append(exp.state)
        if idx > 5:
            break
    assert states == [0, 1, 2, 3, 0, 1, 2]


def test_replaybuffer(car_env):
    source = experience.ExperienceSource(car_env, agent=DummyAgent())
    buf = experience.ExperienceReplayBuffer(source, buffer_size=2)
    assert len(buf) == 0
    assert list(buf) == []

    buf.populate(1)
    assert len(buf) == 1
    buf.populate(2)
    assert len(buf) == 2

    buf = experience.ExperienceReplayBuffer(source, buffer_size=10)
    buf.populate(10)
    b = buf.sample(4)
    assert len(b) == 4

    buf_ids = list(map(id, buf))
    check = list(map(lambda v: id(v) in buf_ids, b))
    assert all(check)


def test_vecsync_exp_simple():
    env = gym.vector.SyncVectorEnv([
        lambda: gym.make("MountainCar-v0"),
        lambda: gym.make("MountainCar-v0"),
    ])
    exp = experience.VectorExperienceSourceFirstLast(
        env, DummyAgent(), gamma=1, steps_count=1,
        env_seed=42)
    e = next(iter(exp))
    assert isinstance(e, list)
    assert len(e) == 2
    assert e[0].reward == -2
    assert e[0].state == approx(np.array([-0.4452088, 0]))
    assert e[0].last_state == approx(np.array([-0.4499448, -0.00315349]))


def test_vector_rewards():
    q = collections.deque()
    q.append(np.array([1, 2, 3], dtype=np.float32))
    q.append(np.array([2, 3, 4], dtype=np.float32))
    dones = collections.deque()
    dones.append(np.array([False, False, False], dtype=bool))
    dones.append(np.array([False, False, False], dtype=bool))
    r = experience.vector_rewards(q, dones, gamma=1.0)
    assert r == pytest.approx(np.array([3, 5, 7]))

    r = experience.vector_rewards(q, dones, gamma=0.99)
    assert r == pytest.approx(np.array([2.98, 4.97, 6.96]))

    # the same example, but now with episode terminated
    dones.clear()
    dones.append(np.array([False, True,  False], dtype=bool))
    dones.append(np.array([True, False, False], dtype=bool))
    r = experience.vector_rewards(q, dones, gamma=1.0)
    assert r == pytest.approx(np.array([3, 2, 7]))

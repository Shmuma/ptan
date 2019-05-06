import numpy as np
from unittest import TestCase

import gym
from ptan import experience, agent


class DummyAgent(agent.BaseAgent):
    def __call__(self, states, agent_states):
        if isinstance(states, list):
            l = len(states)
        else:
            l = states.shape[0]
        return np.zeros(shape=(l, ), dtype=np.int32), agent_states


class CountingEnv(gym.Env):
    def __init__(self, limit=5):
        self.state = 0
        self.limit = limit
        self.observation_space = gym.spaces.Discrete(limit)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        self.state += 1
        done = self.state == self.limit-1
        reward = self.state
        return self.state, reward, done, {}


class TestExperienceSourceSingleEnv(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("MountainCar-v0")

    def test_one_step(self):
        exp_source = experience.ExperienceSource(self.env, DummyAgent(), steps_count=1)
        for exp in exp_source:
            self.assertEqual(1, len(exp))
            self.assertIsInstance(exp, tuple)
            self.assertIsInstance(exp[0], experience.Experience)
            self.assertAlmostEqual(exp[0].reward, -1.0)
            self.assertFalse(exp[0].done)
            break

    def test_two_steps(self):
        exp_source = experience.ExperienceSource(self.env, DummyAgent(), steps_count=2)
        for exp in exp_source:
            self.assertEqual(2, len(exp))
            break

    def test_short_game(self):
        env = gym.make('CartPole-v0')
        exp_source = experience.ExperienceSource(env, DummyAgent(), steps_count=1)
        for step, exp in enumerate(exp_source):
            self.assertIsInstance(exp, tuple)
            self.assertIsInstance(exp[0], experience.Experience)
            if exp[0].done:
                break


class TestExperienceSourceManyEnv(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = [gym.make("MountainCar-v0") for _ in range(10)]

    def test_one_step(self):
        exp_source = experience.ExperienceSource(self.envs, DummyAgent(), steps_count=1)
        for exp in exp_source:
            self.assertEqual(1, len(exp))
            break


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


class TestExperienceSourceStateful(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = [gym.make("CartPole-v0") for _ in range(2)]

    def test_state(self):
        actions_count = self.envs[0].action_space.n
        my_agent = StatefulAgent(self.envs[0].action_space)
        steps = 3
        exp_source = experience.ExperienceSource(self.envs, my_agent, steps_count=steps)

        for _, exp in zip(range(100), exp_source):
            prev_act = None
            for e in exp:
                if prev_act is not None:
                    self.assertEqual(e.action, (prev_act+1) % actions_count)
                prev_act = e.action
            if len(exp) != steps:
                self.assertTrue(exp[-1].done)


class TestExperienceReplayBuffer(TestCase):
    @classmethod
    def setUpClass(cls):
        env = gym.make("MountainCar-v0")
        cls.source = experience.ExperienceSource(env, agent=DummyAgent())

    def test_len(self):
        buf = experience.ExperienceReplayBuffer(self.source, buffer_size=2)
        self.assertEqual(0, len(buf))
        self.assertEqual([], list(buf))

        buf.populate(1)
        self.assertEqual(1, len(buf))

        buf.populate(2)
        self.assertEqual(2, len(buf))

    def test_sample(self):
        buf = experience.ExperienceReplayBuffer(self.source, buffer_size=10)
        buf.populate(10)
        b = buf.sample(4)
        self.assertEqual(4, len(b))

        buf_ids = list(map(id, buf))
        check = list(map(lambda v: id(v) in buf_ids, b))
        self.assertTrue(all(check))

        b = buf.sample(20)
        self.assertEqual(10, len(b))


class TestUtils(TestCase):
    def test_group_list(self):
        r = experience._group_list([1, 2, 3], [3])
        self.assertEqual(r, [[1, 2, 3]])
        r = experience._group_list([1, 2, 3], [1, 1, 1])
        self.assertEqual(r, [[1], [2], [3]])
        r = experience._group_list([1, 2, 3], [1, 2])
        self.assertEqual(r, [[1], [2, 3]])


class TestExperienceSourceFirstLast(TestCase):
    def test_simple(self):
        env = CountingEnv()
        agent = DummyAgent()
        exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
        states = []
        for idx, exp in enumerate(exp_source):
            states.append(exp.state)
            if idx > 5:
                break
        # NB: there is no last state(4), as we don't record it
        self.assertEquals(states, [0, 1, 2, 3, 0, 1, 2])


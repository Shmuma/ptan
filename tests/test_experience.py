from unittest import TestCase

import gym
from ptan import experience


class TestExperienceSource(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("MountainCar-v0")

    @staticmethod
    def agent_zero(state):
        return 0

    def test_one_step(self):
        exp_source = experience.ExperienceSource(self.env, self.agent_zero, steps_count=1)
        for exp in exp_source:
            self.assertEqual(2, len(exp))
            self.assertIsInstance(exp, tuple)
            self.assertIsInstance(exp[0], experience.Experience)
            self.assertAlmostEqual(exp[0].reward, -1.0)
            self.assertFalse(exp[0].done)
            break

    def test_two_steps(self):
        exp_source = experience.ExperienceSource(self.env, self.agent_zero, steps_count=2)
        for exp in exp_source:
            self.assertEqual(3, len(exp))
            break

    def test_short_game(self):
        env = gym.make('CartPole-v0')
        exp_source = experience.ExperienceSource(env, self.agent_zero, steps_count=1)
        for step, exp in enumerate(exp_source):
            self.assertIsInstance(exp, tuple)
            self.assertIsInstance(exp[0], experience.Experience)
            if exp[0].done:
                self.assertEqual(2, len(exp))
                self.assertIsNotNone(exp[1].state)
                self.assertIsNone(exp[1].reward)
                self.assertIsNone(exp[1].action)
                break
            elif exp[1].done:
                self.assertEqual(2, len(exp))



    pass

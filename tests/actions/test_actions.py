import torch
from torch.autograd import Variable

from unittest import TestCase

from ptan.common.env_params import EnvParams
from ptan.actions.epsilon_greedy import ActionSelectorEpsilonGreedy


class TestActionSelectorEpsilonGreedy(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = EnvParams(n_actions=2, state_shape=(6,))

    # TODO: how to test stochastic functionality?

    def test_deterministic(self):
        selector = ActionSelectorEpsilonGreedy(epsilon=0.0, params=self.params)
        q = torch.Tensor([[1.0, 0.0],
                          [0.0, 1.0],
                          [0.2, 0.8]])
        r = selector(Variable(q))
        self.assertTrue(r.data.eq(torch.LongTensor([0, 1, 1])).all())

    def test_random(self):
        selector = ActionSelectorEpsilonGreedy(epsilon=1.0, params=self.params)
        q = torch.Tensor([[1.0, 0.0],
                          [0.0, 1.0],
                          [0.2, 0.8]])
        r = selector(Variable(q))
        self.assertEqual(r.size(), (3, 1))


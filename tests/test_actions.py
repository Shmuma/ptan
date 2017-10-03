import numpy as np
from unittest import TestCase

import ptan.actions as actions


class TestArgmaxActionSelector(TestCase):
    def test_args(self):
        selector = actions.ArgmaxActionSelector()
        self.assertRaises(AssertionError, lambda: selector(1))
        self.assertRaises(AssertionError, lambda: selector([[1, 1]]))

    def test_simple(self):
        selector = actions.ArgmaxActionSelector()
        np.testing.assert_equal(selector(np.array([[1, 2], [0, -10], [0.01, 0.1]])),
                                [1, 0, 1])


class TestEpsilonGreedyActionSelector(TestCase):
    def test_deterministic(self):
        selector = actions.EpsilonGreedyActionSelector(epsilon=0.0)
        r = selector(np.array(
            [[1.0, 0.0],
             [0.0, 1.0],
             [0.2, 0.8]]))
        np.testing.assert_equal(r, [0, 1, 1])

    def test_random(self):
        selector = actions.EpsilonGreedyActionSelector(epsilon=1.0)
        r = selector(np.array(
            [[1.0, 0.0],
             [0.0, 1.0],
             [0.2, 0.8]]))
        self.assertEqual(r.shape, (3, ))

        # test stochastic
        selector = actions.EpsilonGreedyActionSelector(epsilon=0.5)
        count_total = 0
        count_one = 0
        for _ in range(1000):
            r = selector(np.array(
                [[1.0, 0.0],
                 [1.0, 0.0],
                 [1.0, 0.0],
                 [1.0, 0.0]]))
            count_total += r.shape[0]
            count_one += np.sum(r)
        eps = abs(count_one*4 / count_total - 1.)
        self.assertTrue(eps < 0.1)



import pytest
import numpy as np
import numpy.testing as npt

import ptan.actions as actions


def test_argmax():
    sel = actions.ArgmaxActionSelector()
    with pytest.raises(AssertionError):
        sel(1)
    with pytest.raises(AssertionError):
        sel([1, 2])

    s = np.array([[1, 2], [0, -10], [0.01, 0.1]])
    r = sel(s)
    npt.assert_equal(r, [1, 0, 1])
#
#
# class TestEpsilonGreedyActionSelector(TestCase):
#     def test_deterministic(self):
#         selector = actions.EpsilonGreedyActionSelector(epsilon=0.0)
#         r = selector(np.array(
#             [[1.0, 0.0],
#              [0.0, 1.0],
#              [0.2, 0.8]]))
#         np.testing.assert_equal(r, [0, 1, 1])
#
#     def test_random(self):
#         selector = actions.EpsilonGreedyActionSelector(epsilon=1.0)
#         r = selector(np.array(
#             [[1.0, 0.0],
#              [0.0, 1.0],
#              [0.2, 0.8]]))
#         self.assertEqual(r.shape, (3, ))
#
#         # test stochastic
#         selector = actions.EpsilonGreedyActionSelector(epsilon=0.5)
#         count_total = 0
#         count_one = 0
#         for _ in range(1000):
#             r = selector(np.array(
#                 [[1.0, 0.0],
#                  [1.0, 0.0],
#                  [1.0, 0.0],
#                  [1.0, 0.0]]))
#             count_total += r.shape[0]
#             count_one += np.sum(r)
#         eps = abs(count_one*4 / count_total - 1.)
#         self.assertTrue(eps < 0.1)
#
#

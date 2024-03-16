import collections
import numpy as np
import numpy.testing as npt
import pytest

import ptan.actions as actions


def test_argmax():
    sel = actions.ArgmaxActionSelector()
    s = np.array([[1, 2], [0, -10], [0.01, 0.1]])
    r = sel(s)
    npt.assert_equal(r, [1, 0, 1])


def test_epsilon_deterministic():
    selector = actions.EpsilonGreedyActionSelector(epsilon=0.0)
    s = np.array(
        [[1.0, 0.0],
         [0.0, 1.0],
         [0.2, 0.8]])
    r = selector(s)
    npt.assert_equal(r, [0, 1, 1])

    selector.epsilon = 1.0
    with pytest.raises(ValueError):
        selector.epsilon = 1.1


def test_epsilon_random():
    selector = actions.EpsilonGreedyActionSelector(epsilon=1.0)
    r = selector(np.array(
        [[1.0, 0.0],
         [0.0, 1.0],
         [0.2, 0.8]]))
    assert r.shape == (3, )

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
    assert eps < 0.1


def test_prob_action_sel():
    sel = actions.ProbabilityActionSelector()
    stats = [
        collections.Counter()
        for _ in range(3)
    ]

    for _ in range(1000):
        acts = sel(np.array([
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0]
        ]))
        for idx, act in enumerate(acts):
            stats[idx][act] += 1

    assert 0.7 < stats[0][1] / 1000 < 0.9
    assert stats[1][2] == 1000
    assert stats[2][2] == 0
    assert 0.4 < stats[2][0] / 1000 < 0.6


def test_epsilon_tracker():
    tracker = actions.EpsilonTracker(
        actions.EpsilonGreedyActionSelector(),
        1.0, 0.0,
        100
    )
    assert tracker.selector.epsilon == 1.0
    tracker.frame(10)
    assert round(tracker.selector.epsilon, 1) == 0.9
    tracker.frame(90)
    assert round(tracker.selector.epsilon, 1) == 0.1
    tracker.frame(100)
    assert tracker.selector.epsilon == 0.0
    tracker.frame(110)
    assert tracker.selector.epsilon == 0.0

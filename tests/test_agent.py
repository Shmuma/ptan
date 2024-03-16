from ptan import agent, actions

import numpy as np
import numpy.testing as npt

import torch
from torch import nn


class DQNNet(nn.Module):
    def __init__(self, actions: int):
        super(DQNNet, self).__init__()
        self.actions = actions

    def forward(self, x):
        # we always produce diagonal tensor of shape (batch_size, actions)
        return torch.eye(x.size()[0], self.actions)


def test_dqn_agent():
    net = DQNNet(actions=3)
    selector = actions.ArgmaxActionSelector()
    ag = agent.DQNAgent(model=net, action_selector=selector)
    ag_out, ag_st = ag(np.zeros(shape=(2, 5)))
    assert ag_st == [None, None]
    npt.assert_equal(ag_out, [0, 1])


def test_default_states_preprocessor():
    r = agent.default_states_preprocessor([np.array([1, 2, 3])])
    assert torch.is_tensor(r)
    assert r.shape == (1, 3)
    assert r.dtype == torch.int64

    a = [
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
    ]
    r = agent.default_states_preprocessor(a)
    assert torch.is_tensor(r)
    assert r.shape == (2, 3)


def test_float32_preprocessor():
    r = agent.float32_preprocessor([np.array([1, 2, 3])])
    assert torch.is_tensor(r)
    assert r.shape == (1, 3)
    assert isinstance(r, torch.Tensor)

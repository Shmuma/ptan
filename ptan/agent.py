"""
Agent is something which converts states into actions
"""
from .common import env_params

import numpy as np
import torch
from torch.autograd import Variable


class DQNAgent:
    """
    DQNAgent calculates Q values from states and converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector):
        self.dqn_model = dqn_model
        self.action_selector = action_selector

    def __call__(self, states):
        v = Variable(torch.from_numpy(np.array(states, dtype=np.float32)))
        if env_params.get().cuda_enabled:
            v = v.cuda()
        q = self.dqn_model(v)
        actions = self.action_selector(q)
        return actions.data.cpu().numpy()


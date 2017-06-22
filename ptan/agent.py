"""
Agent is something which converts states into actions
"""
from .common import env_params

import copy
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


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())


class PolicyAgent:
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, states):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        v = Variable(torch.from_numpy(np.array(states, dtype=np.float32)))
        if env_params.get().cuda_enabled:
            v = v.cuda()
        probs = self.model(v).data.cpu().numpy()
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


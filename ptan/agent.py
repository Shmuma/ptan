"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process 
        :param agent_states: list of states with the same length as observations 
        :return: tuple of actions, states 
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values 
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, cuda=False):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.cuda = cuda

    def __call__(self, states, agent_states):
        v = Variable(torch.FloatTensor(states))
        if self.cuda:
            v = v.cuda()
        q_v = self.dqn_model(v)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states


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
    def __init__(self, model, cuda=False, apply_softmax=False):
        self.model = model
        self.cuda = cuda
        self.apply_softmax = apply_softmax

    def __call__(self, states):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        v = Variable(torch.from_numpy(np.array(states, dtype=np.float32)))
        if self.cuda:
            v = v.cuda()
        probs_v = self.model(v)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v)
        probs = probs_v.data.cpu().numpy()
        actions = []
        for prob in probs:
            actions.append([np.random.choice(len(prob), p=prob)])
        return np.array(actions)


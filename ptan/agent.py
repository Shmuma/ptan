"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from . import actions


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


def default_states_preprocessor(states, fsa=False):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if fsa:
        if len(states) == 1:
            im_states = np.expand_dims(states[0]['image'], 0)
            lo_states = np.expand_dims(np.vstack(states[0]['logic']), 0)
        else:
            im_states = np.array([np.array(s, copy=False) for s['image'] in states], copy=False)
            lo_states = np.array([np.vstack(s, copy=False) for s['logic'] in states], copy=False)
        return {'image': torch.tensor(im_states), 'logic': torch.tensor(lo_states)}
    else:
        if len(states) == 1:
            np_states = np.expand_dims(states[0], 0)
        else:
            np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
        return torch.tensor(np_states)


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, device="cpu", fsa=False,
                 preprocessor=default_states_preprocessor, epsilon_tracker=None):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device
        self.fsa = fsa
        self.epsilon_tracker = epsilon_tracker

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states, self.fsa)
            if torch.is_tensor(states):
                states = states.to(self.device)
            elif self.fsa:
                if torch.is_tensor(states['image']):
                    states['image'] = states['image'].to(self.device)
                if torch.is_tensor(states['logic']):
                    states['logic'] = states['logic'].to(self.device)
        if self.dqn_model.__class__.__name__ == 'FSADQNATTNMatching' or self.dqn_model.__class__.__name__ == 'FSADQNATTNMatchingFC':
            q_v, recon, conv_output = self.dqn_model(states)
        elif self.dqn_model.__class__.__name__ == 'FSADQNAppendToFCL1Conv':
            q_v, conv_out = self.dqn_model(states)
        else:
            q_v = self.dqn_model(states)
        if self.epsilon_tracker and self.fsa:
                self.epsilon_tracker.frame(tuple(states['logic'][0][-1].cpu().numpy()))
        q = q_v.data.cpu().numpy()
        if self.fsa:
            actions = self.action_selector(q, tuple(states['logic'][0][-1].cpu().numpy()))
        else:
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

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class PolicyAgent(BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    # TODO: unify code with DQNAgent, as only action selector is differs.
    def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
                 fsa=False, apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor
        self.fsa = fsa

    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states, self.fsa)
            if torch.is_tensor(states):
                states = states.to(self.device)
            elif self.fsa:
                if torch.is_tensor(states['image']):
                    states['image'] = states['image'].to(self.device)
                if torch.is_tensor(states['logic']):
                    states['logic'] = states['logic'].to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states

class ActorCriticAgent(BaseAgent):
    """
    Policy agent which returns policy and value tensors from observations. Value are stored in agent's state
    and could be reused for rollouts calculations by ExperienceSource.
    """
    def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v, values_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        agent_states = values_v.data.squeeze().cpu().numpy().tolist()
        return np.array(actions), agent_states

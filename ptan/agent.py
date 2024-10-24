"""
Agent is something which converts states into actions and has state
"""
import abc
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import typing as tt

from . import actions

States = tt.List[np.ndarray] | np.ndarray
AgentStates = tt.List[tt.Any]
Preprocessor = tt.Callable[[States], torch.Tensor]

CPU_DEVICE = torch.device("cpu")


def default_states_preprocessor(states: States) -> torch.Tensor:
    """
    Convert list of states into the form suitable for model
    :param states: list of numpy arrays with states or numpy array
    :return: torch.Tensor
    """
    if isinstance(states, list):
        if len(states) == 1:
            np_states = np.expand_dims(states[0], 0)
        else:
            np_states = np.asarray([np.asarray(s) for s in states])
    else:
        np_states = states
    return torch.as_tensor(np_states)


def float32_preprocessor(states: States):
    np_states = np.array(states, dtype=np.float32)
    return torch.as_tensor(np_states)


class BaseAgent(abc.ABC):
    """
    Base Agent, sharing most of logic with concrete agent implementations.
    """
    def initial_state(self) -> tt.Optional[tt.Any]:
        """
        Should create initial empty state for the agent. It will be
        called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    @abc.abstractmethod
    def __call__(self, states: States, agent_states: AgentStates) -> tt.Tuple[np.ndarray, AgentStates]:
        ...


class NNAgent(BaseAgent):
    """
    Network-based agent
    """
    def __init__(self, model: nn.Module, action_selector: actions.ActionSelector,
                 device: torch.device, preprocessor: Preprocessor):
        """
        Constructor of base agent
        :param model: model to be used
        :param action_selector: action selector
        :param device: device for tensors
        :param preprocessor: states preprocessor
        """
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.preprocessor = preprocessor

    @abc.abstractmethod
    def _net_filter(self, net_out: tt.Any, agent_states: AgentStates) -> \
            tt.Tuple[torch.Tensor, AgentStates]:
        """
        Internal method, processing network output and states into selector's input and new states
        :param net_out: output from the network
        :param agent_states: agent states
        :return: tuple with tensor to be fed into selector and new states
        """
        ...

    @torch.no_grad()
    def __call__(self, states: States, agent_states: AgentStates = None) -> tt.Tuple[np.ndarray, AgentStates]:
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        q_v = self.model(states)
        q_v, new_states = self._net_filter(q_v, agent_states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, new_states


class DQNAgent(NNAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, model: nn.Module, action_selector: actions.ActionSelector,
                 device: torch.device = CPU_DEVICE,
                 preprocessor: Preprocessor = default_states_preprocessor):
        super().__init__(model, action_selector=action_selector, device=device,
                         preprocessor=preprocessor)

    # not needed in DQN - we don't process Q-values returned
    def _net_filter(self, net_out: tt.Any, agent_states: AgentStates) -> \
            tt.Tuple[torch.Tensor, AgentStates]:
        assert torch.is_tensor(net_out)
        return net_out, agent_states


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model: nn.Module):
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


class PolicyAgent(NNAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    def __init__(self, model: nn.Module,
                 action_selector: actions.ActionSelector = actions.ProbabilityActionSelector(),
                 device: torch.device = CPU_DEVICE, apply_softmax: bool = False,
                 preprocessor: Preprocessor = default_states_preprocessor):
        super().__init__(model=model, action_selector=action_selector, device=device,
                         preprocessor=preprocessor)
        self.apply_softmax = apply_softmax

    def _net_filter(self, net_out: tt.Any, agent_states: AgentStates) -> \
            tt.Tuple[torch.Tensor, AgentStates]:
        assert torch.is_tensor(net_out)
        if self.apply_softmax:
            return F.softmax(net_out, dim=1), agent_states
        return net_out, agent_states


class ActorCriticAgent(NNAgent):
    """
    Policy agent which returns policy and value tensors from observations. Value are stored in agent's state
    and could be reused for rollouts calculations by ExperienceSource.
    """
    def __init__(self, model: nn.Module,
                 action_selector: actions.ActionSelector = actions.ProbabilityActionSelector(),
                 device: torch.device = CPU_DEVICE, apply_softmax: bool = False,
                 preprocessor: Preprocessor = default_states_preprocessor):
        super().__init__(model=model, action_selector=action_selector, device=device, preprocessor=preprocessor)
        self.apply_softmax = apply_softmax

    def _net_filter(self, net_out: tt.Any, agent_states: AgentStates) -> \
            tt.Tuple[torch.Tensor, AgentStates]:
        assert isinstance(net_out, tuple)
        policy_v, values_v = net_out
        assert torch.is_tensor(policy_v)
        assert torch.is_tensor(values_v)
        agent_states = values_v.data.squeeze().cpu().numpy().tolist()
        if self.apply_softmax:
            return F.softmax(policy_v, dim=1), agent_states
        return policy_v, agent_states

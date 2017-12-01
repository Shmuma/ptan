import gym
import torch
import random
import collections
from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque

from .agent import BaseAgent
from .common import utils

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.agent_states = [agent.initial_state() for _ in self.pool]

    def __iter__(self):
        states, histories, cur_rewards, cur_steps = [], [], [], []
        for env in self.pool:
            states.append(env.reset())
            histories.append(deque(maxlen=self.steps_count))
            cur_rewards.append(0.0)
            cur_steps.append(0)

        iter_idx = 0
        while True:
            actions, self.agent_states = self.agent(states, self.agent_states)

            for idx, env in enumerate(self.pool):
                state = states[idx]
                action = actions[idx]
                history = histories[idx]
                next_state, r, is_done, _ = env.step(action)
                cur_rewards[idx] += r
                cur_steps[idx] += 1
                history.append(Experience(state=state, action=action, reward=r, done=is_done))
                if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                    yield tuple(history)
                states[idx] = next_state
                if is_done:
                    # generate tail of history
                    while len(history) >= 1:
                        yield tuple(history)
                        history.popleft()
                    self.total_rewards.append(cur_rewards[idx])
                    self.total_steps.append(cur_steps[idx])
                    cur_rewards[idx] = 0.0
                    cur_steps[idx] = 0
                    states[idx] = env.reset()
                    self.agent_states[idx] = self.agent.initial_state()
                    history.clear()
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


class ExperienceSourceBuffer:
    """
    The same as ExperienceSource, but takes episodes from the buffer
    """
    def __init__(self, buffer, steps_count=1):
        """
        Create buffered experience source
        :param buffer: list of episodes, each is a list of Experience object
        :param steps_count: count of steps in every entry
        """
        self.update_buffer(buffer)
        self.steps_count = steps_count

    def update_buffer(self, buffer):
        self.buffer = buffer
        self.lens = list(map(len, buffer))

    def __iter__(self):
        """
        Infinitely sample episode from the buffer and then sample item offset
        """
        while True:
            episode = random.randrange(len(self.buffer))
            ofs = random.randrange(self.lens[episode] - self.steps_count - 1)
            yield self.buffer[episode][ofs:ofs+self.steps_count]


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

class PrioReplayBufferNaive:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, experience_source, buffer_size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(experience_source, buffer_size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = utils.SumSegmentTree(it_capacity)
        self._it_min = utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        return samples, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class BatchPreprocessor:
    """
    Abstract preprocessor class descendants to which converts experience
    batch to form suitable to learning.
    """
    def preprocess(self, batch):
        raise NotImplementedError


class QLearningPreprocessor(BatchPreprocessor):
    """
    Supports SimpleDQN, TargetDQN, DoubleDQN and can additionally feed TD-error back to
    experience replay buffer.

    To use different modes, use appropriate class method
    """
    def __init__(self, model, target_model, use_double_dqn=False, batch_td_error_hook=None, gamma=0.99, cuda=False):
        self.model = model
        self.target_model = target_model
        self.use_double_dqn = use_double_dqn
        self.batch_dt_error_hook = batch_td_error_hook
        self.gamma = gamma
        self.cuda = cuda

    @staticmethod
    def simple_dqn(model, **kwargs):
        return QLearningPreprocessor(model=model, target_model=None, use_double_dqn=False, **kwargs)

    @staticmethod
    def target_dqn(model, target_model, **kwards):
        return QLearningPreprocessor(model, target_model, use_double_dqn=False, **kwards)

    @staticmethod
    def double_dqn(model, target_model, **kwargs):
        return QLearningPreprocessor(model, target_model, use_double_dqn=True, **kwargs)

    def _calc_Q(self, states_first, states_last):
        """
        Calculates apropriate q values for first and last states. Way of calculate depends on our settings.
        :param states_first: numpy array of first states
        :param states_last: numpy array of last states
        :return: tuple of numpy arrays of q values
        """
        # here we need both first and last values calculated using our main model, so we
        # combine both states into one batch for efficiency and separate results later
        if self.target_model is None or self.use_double_dqn:
            states_t = torch.from_numpy(np.concatenate((states_first, states_last), axis=0))
            states_v = Variable(states_t)
            if self.cuda:
                states_v = states_v.cuda()
            res_both = self.model(states_v).data.cpu().numpy()
            return res_both[:len(states_first)], res_both[len(states_first):]

        # in this case we have target_model set and use_double_dqn==False
        # so, we should calculate first_q and last_q using different models
        states_first_v = Variable(torch.from_numpy(states_first))
        states_last_v = Variable(torch.from_numpy(states_last))
        if self.cuda:
            states_first_v = states_first_v.cuda()
            states_last_v = states_last_v.cuda()
        q_first = self.model(states_first_v).data
        q_last = self.target_model(states_last_v).data
        return q_first.cpu().numpy(), q_last.cpu().numpy()

    def _calc_target_rewards(self, states_last, q_last):
        """
        Calculate rewards from final states according to variants from our construction:
        1. simple DQN: max(Q(states, model))
        2. target DQN: max(Q(states, target_model))
        3. double DQN: Q(states, target_model)[argmax(Q(states, model)]
        :param states_last: numpy array of last states from the games
        :param q_last: numpy array of last q values
        :return: vector of target rewards
        """
        # in this case we handle both simple DQN and target DQN
        if self.target_model is None or not self.use_double_dqn:
            return q_last.max(axis=1)

        # here we have target_model set and use_double_dqn==True
        actions = q_last.argmax(axis=1)
        # calculate Q values using target net
        states_last_v = Variable(torch.from_numpy(states_last))
        if self.cuda:
            states_last_v = states_last_v.cuda()
        q_last_target = self.target_model(states_last_v).data.cpu().numpy()
        return q_last_target[range(q_last_target.shape[0]), actions]

    def preprocess(self, batch):
        """
        Calculates data for Q learning from batch of observations
        :param batch: list of lists of Experience objects
        :return: tuple of numpy arrays:
            1. states -- observations
            2. target Q-values
            3. vector of td errors for every batch entry
        """
        # first and last states for every entry
        state_0 = np.array([exp[0].state for exp in batch], dtype=np.float32)
        state_L = np.array([exp[-1].state for exp in batch], dtype=np.float32)

        q0, qL = self._calc_Q(state_0, state_L)
        rewards = self._calc_target_rewards(state_L, qL)

        td = np.zeros(shape=(len(batch),))

        for idx, (total_reward, exps) in enumerate(zip(rewards, batch)):
            # game is done, no final reward
            if exps[-1].done:
                total_reward = 0.0
            for exp in reversed(exps[:-1]):
                total_reward *= self.gamma
                total_reward += exp.reward
            # update total reward and calculate td error
            act = exps[0].action
            td[idx] = q0[idx][act] - total_reward
            q0[idx][act] = total_reward

        return state_0, q0, td

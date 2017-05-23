import torch
from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque, OrderedDict

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments
    
    Every experience contains n+1 list of Experience entries
    """
    def __init__(self, env, agent, steps_count=1):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        """
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.total_rewards = []

    def __iter__(self):
        states, histories, cur_rewards = [], [], []
        for env in self.pool:
            states.append(env.reset())
            histories.append(deque())
            cur_rewards.append(0.0)

        while True:
            actions = self.agent(np.array(states))

            for idx, env in enumerate(self.pool):
                state = states[idx]
                action = actions[idx][0]
                history = histories[idx]
                next_state, r, is_done, _ = env.step(action)
                cur_rewards[idx] += r
                history.append(Experience(state=state, action=action, reward=r, done=is_done))
                while len(history) > self.steps_count+1:
                    history.popleft()
                if len(history) == self.steps_count+1:
                    yield tuple(history)
                states[idx] = next_state
                if is_done:
                    if len(history) > self.steps_count+1:
                        history.popleft()
                    # generate tail of history
                    while len(history) >= 1:
                        yield tuple(history)
                        history.popleft()
                    self.total_rewards.append(cur_rewards[idx])
                    cur_rewards[idx] = 0.0
                    states[idx] = env.reset()
                    history.clear()

    def pop_total_rewards(self):
        r = self.total_rewards
        self.total_rewards = []
        return r


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size=None):
        self.buffer_size = buffer_size
        self.experience_source = experience_source
        self.experience_source_iter = iter(experience_source)
        self.buffer = deque()

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
            return list(self.buffer)
        keys = np.random.choice(range(len(self.buffer)), batch_size, replace=False)
        return [self.buffer[key] for key in keys]

    def batches(self, batch_size):
        """
        Iterate batches of given size once (i.e. one epoch over buffer)
        :param batch_size: 
        """
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        while samples > 0:
            entry = next(self.experience_source_iter)
            self.buffer.append(entry)
            samples -= 1
        if self.buffer_size is not None:
            while len(self.buffer) > self.buffer_size:
                self.buffer.popleft()


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
    def __init__(self, model, target_model, use_double_dqn=False, batch_td_error_hook=None, gamma=0.99):
        self.model = model
        self.target_model = target_model
        self.use_double_dqn = use_double_dqn
        self.batch_dt_error_hook = batch_td_error_hook
        self.gamma = gamma

    @staticmethod
    def simple_dqn(model, **kwargs):
        return QLearningPreprocessor(model=model, target_model=None, use_double_dqn=False, **kwargs)

    @staticmethod
    def target_dqn(model, target_model, **kwards):
        return QLearningPreprocessor(model, target_model, use_double_dqn=False, **kwards)

    @staticmethod
    def double_dqn(model, target_model, **kwargs):
        return QLearningPreprocessor(model, target_model, use_double_dqn=True, **kwargs)

    def _calc_Q(self, states):
        # calculate Q values from basic model
        states_t = torch.from_numpy(np.array(states, dtype=np.float32))
        states_v = Variable(states_t).cuda()
        return self.model(states_v).data

    def _calc_target_rewards(self, states):
        """
        Calculate rewards from final states according to variants from our construction:
        1. simple DQN: max(Q(states, model))
        2. target DQN: max(Q(states, target_model))
        3. double DQN: Q(states, target_model)[argmax(Q(states, model)]
        :param states: 
        :return: 
        """
        states_t = torch.from_numpy(np.array(states, dtype=np.float32))
        states_v = Variable(states_t).cuda()

        # simple DQN case
        if self.target_model is None:
            q = self.model(states_v)
            return q.data.max(1)[0].squeeze().cpu().numpy()

        # have target net, but no DDQN
        if not self.use_double_dqn:
            pass

    def preprocess(self, batch):
        # first and last states for every entry
        state_0 = np.array([exp[0].state for exp in batch], dtype=np.float32)
        state_L = np.array([exp[-1].state for exp in batch], dtype=np.float32)

        q0 = self._calc_Q(state_0)
        rewards = self._calc_target_rewards(state_L)

        for idx, (total_reward, exps) in enumerate(zip(rewards, batch)):
            # game is done, no final resward
            if exps[-1].done:
                total_reward = 0.0
            for exp in reversed(exps[:-1]):
                total_reward *= self.gamma
                total_reward += exp.reward
            # update total reward
            # TODO: here we should calculate TD-error
            q0[idx][exps[0].action] = total_reward

        return torch.from_numpy(state_0), q0

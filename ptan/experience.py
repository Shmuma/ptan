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

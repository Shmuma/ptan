import numpy as np

from collections import namedtuple, deque, OrderedDict

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using single environment
    
    Every experience contains n+1 list of Experience entries
    """
    def __init__(self, env, agent, steps_count=1):
        """
        Create simple experience source
        :param env: environment to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        """
        self.env = env
        self.agent = agent
        self.steps_count = steps_count
        self.total_rewards = []

    def __iter__(self):
        state = self.env.reset()
        history = deque()
        total_reward = 0.0
        while True:
            action = self.agent(np.expand_dims(state, axis=0))[0][0]
            next_state, r, is_done, _ = self.env.step(action)
            total_reward += r
            history.append(Experience(state=state, action=action, reward=r, done=is_done))
            if len(history) > self.steps_count+1:
                history.popleft()
            if len(history) == self.steps_count+1:
                yield tuple(history)
            state = next_state
            if is_done:
#                history.append(Experience(state=state, action=None, reward=None, done=None))
                if len(history) > self.steps_count+1:
                    history.popleft()
                # generate tail of history
                while len(history) > 1:
                    yield tuple(history)
                    history.popleft()
                self.total_rewards.append(total_reward)
                total_reward = 0.0
                state = self.env.reset()
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

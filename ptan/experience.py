from collections import namedtuple, deque

from .common import env_params


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
        :param agent: callable to convert state into action to take
        :param steps_count: count of steps to track for every experience chain
        """
        self.env = env
        self.agent = agent
        self.steps_count = steps_count

    def __iter__(self):
        state = self.env.reset()
        history = deque()
        while True:
            action = self.agent(state)
            next_state, r, is_done, _ = self.env.step(action)
            history.append(Experience(state=state, action=action, reward=r, done=is_done))
            if len(history) > self.steps_count+1:
                history.popleft()
            if len(history) == self.steps_count+1:
                yield tuple(history)
            state = next_state
            if is_done:
                history.append(Experience(state=state, action=None, reward=None, done=None))
                if len(history) > self.steps_count+1:
                    history.popleft()
                yield tuple(history)
                state = self.env.reset()
                history = []

class ExperienceReplayBuffer:
    def __init__(self, experience_source, params=env_params.get(), buffer_size=None):
        self.buffer_size = buffer_size
        self.experience_source = experience_source
        self.params = params

        pass


    pass

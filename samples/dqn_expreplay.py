import numpy as np

from ptan.common import env_params
from ptan.actions.epsilon_greedy import ActionSelectorEpsilonGreedy
from ptan import experience

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gym


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    params = env_params.EnvParams.from_env(env)
    env_params.register(params)

    model = nn.Sequential(
        nn.Linear(params.state_shape[0], 50),
        nn.ReLU(),
        nn.Linear(50, params.n_actions)
    )

    loss_fn = nn.MSELoss(size_average=False)

    action_selector = ActionSelectorEpsilonGreedy(epsilon=0.05, params=params)

    def agent(states):
        """
        Return actions to take by a batch of states
        :param states: numpy array with states 
        :return: 
        """
        v = Variable(torch.from_numpy(np.array(states, dtype=np.float32)))
        q = model(v)
        actions = action_selector(q)
        return actions.data.numpy()

    #
    # test_s = Variable(torch.from_numpy(np.array([env.reset()], dtype=np.float32)))
    # print(model(test_s))
    # print(action_selector(model(test_s)))
    # print(loss_fn(model(test_s), Variable(torch.Tensor([[1.0, 0.0, 2.0]]))))

    exp_source = experience.ExperienceSource(env=env, agent=agent, steps_count=1)
    exp_replay = experience.ExperienceReplayBuffer(exp_source, buffer_size=100)

    # populate buffer
    exp_replay.populate(50)
    # sample batch from buffer
    batch = exp_replay.sample(50)
    # convert experience batch into training samples
    

    pass

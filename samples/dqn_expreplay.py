import numpy as np

import ptan
from ptan.common import env_params
from ptan.actions.epsilon_greedy import ActionSelectorEpsilonGreedy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gym


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    params = env_params.EnvParams(env)
    env_params.register(params)

    model = nn.Sequential(
        nn.Linear(params.state_shape[0], 50),
        nn.ReLU(),
        nn.Linear(50, params.n_actions)
    )

    loss_fn = nn.MSELoss(size_average=False)

    action_selector = ActionSelectorEpsilonGreedy(epsilon=0.05, params=params)

    test_s = Variable(torch.from_numpy(np.array([env.reset()], dtype=np.float32)))
    print(model(test_s))
    print(action_selector(model(test_s)))
    print(loss_fn(model(test_s), Variable(torch.Tensor([[1.0, 0.0, 2.0]]))))
    
    pass

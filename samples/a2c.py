#!/usr/bin/env python
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import ptan
from ptan.common import runfile, env_params

import gym

GAMMA = 0.99


class Model(nn.Module):
    def __init__(self, n_actions, input_len):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(input_len, 50)
        self.fc2 = nn.Linear(50, 50)
        self.out_policy = nn.Linear(50, n_actions)
        self.out_value = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        policy = F.softmax(self.out_policy(x))
        value = self.out_value(x)
        return policy, value



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runfile", required=True, help="Name of the runfile to use")
    parser.add_argument("-m", "--monitor", help="Use monitor and save it's data into given dir")
    args = parser.parse_args()

    run = runfile.RunFile(args.runfile)

    cuda_enabled = run.getboolean("defaults", "cuda", fallback=False)
    env = gym.make(run.get("defaults", "env")).env
    if args.monitor:
        env = gym.wrappers.Monitor(env, args.monitor)

    params = env_params.EnvParams.from_env(env)
    env_params.register(params)

    # model returns probability of actions
    model = Model(params.n_actions, params.state_shape[0])
    if cuda_enabled:
        model.cuda()

    v = Variable(torch.from_numpy(np.array([env.reset()], dtype=np.float32)))
    res = model(v)
    print(res)
    pass

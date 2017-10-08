#!/usr/bin/env python3
import gym
import ptan
import argparse
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

log = gym.logger


class ConvNet(nn.Module):
    def __init__(self, input_shape, output_size):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64*6*6, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class PolicyNet(nn.Module):
    def __init__(self, input_size, actions_n):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(input_size, actions_n)

    def forward(self, x):
        return self.fc(x)


class ValueNet(nn.Module):
    def __init__(self, input_size):
        super(ValueNet, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runfile", required=True, help="Name of the runfile to use")
    parser.add_argument("-m", "--monitor", help="Use monitor and save it's data into given dir")
    args = parser.parse_args()

    run = ptan.common.runfile.RunFile(args.runfile)

    cuda_enabled = run.getboolean("defaults", "cuda", fallback=False)

    def make_env():
        env = ptan.common.wrappers.PreprocessImage(gym.make(run.get("defaults", "env")))
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.monitor)
        return env

    env = make_env()
    hidden_size = run['learning'].getint('hidden_size')
    conv_net = ConvNet(env.observation_space.shape, hidden_size)
    value_net = ValueNet(hidden_size)
    policy_net = PolicyNet(hidden_size, env.action_space.n)

    params = itertools.chain(conv_net.parameters(), value_net.parameters(), policy_net.parameters())
    optimizer = optim.RMSprop(params, lr=run['learning'].getfloat('lr'))

    t = Variable(torch.from_numpy(np.expand_dims(env.reset(), 0)))
    o = conv_net(t)
    o = policy_net(o)
    print(o.size())

    agent = ptan.agent.PolicyAgent(model=nn.Sequential(conv_net, policy_net), cuda=cuda_enabled, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSource([make_env() for _ in range(run.getint('defaults', 'games'))],
                                                  agent=agent, steps_count=run.getint('defaults', 'steps'))

    for exp in exp_source:
        print(exp)
        break
    pass

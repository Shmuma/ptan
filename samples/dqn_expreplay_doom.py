#!/usr/bin/env python
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
import ppaquette_gym_doom
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

from ptan.common import runfile, env_params


class Net(nn.Module):
    def __init__(self, n_actions, input_shape=(3, 480, 640)):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.conv4 = nn.Conv2d(64, 64, 2)

        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_size, 50)
        self.fc2 = nn.Linear(50, n_actions)

    def _get_conv_output(self, shape):
        input = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runfile", required=True, help="Name of the runfile to use")
    parser.add_argument("-m", "--monitor", help="Use monitor and save it's data into given dir")
    args = parser.parse_args()

    run = runfile.RunFile(args.runfile)

    cuda_enabled = run.getboolean("defaults", "cuda", fallback=False)
    make_env = lambda: SkipWrapper(4)(ToDiscrete("minimal")(gym.make(run.get("defaults", "env"))))
    env = make_env()

    params = env_params.EnvParams.from_env(env)
    env_params.register(params)

    model = Net(params.n_actions)

    r = env.reset()
    print(r.shape)

    t = torch.from_numpy(np.array([r], dtype=np.float32)).permute(0, 3, 1, 2)
    v = Variable(t)
    q = model(v)
    print(q.data)
    pass

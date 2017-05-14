#!/usr/bin/env python
import os
import time
import argparse

import numpy as np
from scipy.misc import imresize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
from gym.core import ObservationWrapper
from gym.spaces import Box

from ptan.common import runfile, env_params, utils, wrappers
from ptan.actions.epsilon_greedy import ActionSelectorEpsilonGreedy
from ptan import experience, agent

GAMMA = 0.99

REPORT_ITERS = 10


class PreprocessAtari(ObservationWrapper):
    def __init__(self, env, target_shape):
        ObservationWrapper.__init__(self, env)

        self.target_shape = target_shape
        self.observation_space = Box(0.0, 1.0, target_shape)

    def _observation(self, img):
        gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
        return (imresize(gray, self.img_size)/255.0).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runfile", required=True, help="Name of the runfile to use")
    parser.add_argument("-m", "--monitor", help="Use monitor and save it's data into given dir")
    parser.add_argument("-s", "--save", help="Directory to save model state")
    args = parser.parse_args()

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)

    run = runfile.RunFile(args.runfile)

    grayscale = run.getboolean("defaults", "grayscale", fallback=True)
    im_width = run.getint("defaults", "image_width", fallback=80)
    im_height = run.getint("defaults", "image_height", fallback=80)

    def make_env():
        e = wrappers.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make(run.get("defaults", "env")))),
                                     width=im_width, height=im_height, grayscale=grayscale)
        if args.monitor:
            e = gym.wrappers.Monitor(e, args.monitor)
        return e

    pass

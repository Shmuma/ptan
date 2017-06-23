#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import ptan
from ptan.common import runfile, env_params

import gym


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
    model = nn.Sequential(
        nn.Linear(params.state_shape[0], 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, params.n_actions),
        nn.Softmax()
    )
    if cuda_enabled:
        model.cuda()

    agent = ptan.agent.PolicyAgent(model)
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=run.getint("defaults", "n_steps"))
    exp_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, run.getint("exp_buffer", "size"))

    optimizer = optim.Adam(model.parameters(), lr=run.getfloat("learning", "lr"))

    def calc_loss(batch):
        """
        Calculate loss expression from data batch
        :param batch: batch data 
        :return: loss tensor
        """
        result = torch.zero()
        for exp in batch:
            # calculate total reward
            probs = model(exp[0].state)

    while True:
        exp_buffer.populate(run.getint("exp_buffer", "populate"))
        batch = exp_buffer.sample(run.getint("learning", "batch_size"))

    pass

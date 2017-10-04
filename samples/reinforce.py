#!/usr/bin/env python
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import ptan
from ptan.common import runfile

import gym

GAMMA = 0.99


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

    # model returns probability of actions
    model = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 50),
        nn.ReLU(),
        # nn.Linear(100, 50),
        # nn.ReLU(),
        nn.Linear(50, env.action_space.n),
        nn.Softmax()
    )
    if cuda_enabled:
        model.cuda()

    agent = ptan.agent.PolicyAgent(model, cuda=cuda_enabled)
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=run.getint("defaults", "n_steps"))
    exp_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, run.getint("exp_buffer", "size"))

    optimizer = optim.Adam(model.parameters(), lr=run.getfloat("learning", "lr"))

    def calc_loss(batch):
        """
        Calculate loss expression from data batch
        :param batch: batch data
        :return: loss tensor
        """
        entropy_beta = run.getfloat("defaults", "entropy_beta", fallback=0.0)
        result = Variable(torch.FloatTensor(1).zero_())
        for exps in batch:
            # calculate total reward
            R = 0
            for exp in reversed(exps):
                R *= GAMMA
                R += exp.reward

            v = Variable(torch.from_numpy(np.array([exps[0].state], dtype=np.float32)))
            probs = model(v)[0]
            lp = probs.log()
            entropy = torch.sum(torch.mul(lp, probs))
            result += entropy_beta * entropy
            prob = probs[exps[0].action]
            result += -prob.log() * R
        return result / len(batch)

    iter = 0
    min_iters = run.getint("defaults", "min_iters", fallback=0)
    while True:
        exp_buffer.populate(run.getint("exp_buffer", "populate"))
        #        batch = exp_buffer.sample(run.getint("learning", "batch_size"))
        losses = []
        for batch in exp_buffer.batches(run.getint("learning", "batch_size")):
            optimizer.zero_grad()
            loss = calc_loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().numpy()[0])

        total_rewards = exp_source.pop_total_rewards()
        mean_reward = np.mean(total_rewards) if total_rewards else 0.0

        print("%d: mean reward %.3f in %d games, loss %.3f" % (iter, mean_reward, len(total_rewards), np.mean(losses)))
        iter += 1
        if iter > min_iters and mean_reward > 300:
            break

    pass

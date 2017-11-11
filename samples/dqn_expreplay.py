#!/usr/bin/env python
import argparse
import numpy as np

from ptan.common import runfile, utils
from ptan.actions import EpsilonGreedyActionSelector
import ptan

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

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

    model = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 50),
        nn.ReLU(),
        # nn.Linear(100, 100),
        # nn.ReLU(),
        nn.Linear(50, env.action_space.n)
    )
    if cuda_enabled:
        model.cuda()

    loss_fn = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=run.getfloat("learning", "lr"))

    action_selector = EpsilonGreedyActionSelector(epsilon=run.getfloat("defaults", "epsilon"))
    agent = ptan.agent.DQNAgent(model, action_selector, cuda=cuda_enabled)

    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=run.getint("defaults", "n_steps"))
    exp_replay = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=run.getint("exp_buffer", "size"))

    def batch_to_train(batch):
        """
        Convert batch into training data using bellman's equation
        :param batch: list of tuples with Experience instances
        :return:
        """
        states = []
        q_vals = []
        for exps in batch:
            # calculate q_values for first and last state in experience sequence
            # first is needed for reference, last is used to approximate rest value
            v = Variable(torch.from_numpy(np.array([exps[0].state, exps[-1].state], dtype=np.float32)))
            if cuda_enabled:
                v = v.cuda()
            q = model(v)
            # accumulate total reward for the chain
            total_reward = 0.0 if exps[-1].done else q[1].data.max()
            for exp in reversed(exps[:-1]):
                total_reward = exp.reward + GAMMA * total_reward
            train_state = exps[0].state
            train_q = q[0].data
            train_q[exps[0].action] = total_reward
            states.append(train_state)
            q_vals.append(train_q)
        return torch.from_numpy(np.array(states, dtype=np.float32)), torch.stack(q_vals)

    reward_sma = utils.SMAQueue(run.getint("stop", "mean_games", fallback=100))

    for idx in range(10000):
        exp_replay.populate(run.getint("exp_buffer", "populate"))

        for batch in exp_replay.batches(run.getint("learning", "batch_size")):
            optimizer.zero_grad()

            # populate buffer
            states, q_vals = batch_to_train(batch)
            # ready to train
            states, q_vals = Variable(states), Variable(q_vals)
            if cuda_enabled:
                states = states.cuda()
                q_vals = q_vals.cuda()
            l = loss_fn(model(states), q_vals)
            l.backward()
            optimizer.step()

        action_selector.epsilon *= run.getfloat("defaults", "epsilon_decay")

        if idx % 10 == 0:
            total_rewards = exp_source.pop_total_rewards()
            reward_sma += total_rewards
            mean_reward = reward_sma.mean()
            mean_reward_str = "%.2f" % mean_reward if mean_reward is not None else 'None'
            print("%d: Mean reward: %s, done: %d, epsilon: %.4f" % (
                idx, mean_reward_str, len(total_rewards), action_selector.epsilon
            ))

            if run.has_option("stop", "mean_reward") and mean_reward is not None:
                if mean_reward >= run.getfloat("stop", "mean_reward"):
                    print("We've reached mean reward bound, exit")
                    break
    env.close()
    pass

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

import bokeh
from bokeh.plotting import figure
from bokeh.io import output_file, show

GAMMA = 0.99


class Model(nn.Module):
    def __init__(self, n_actions, input_len):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(input_len, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out_policy = nn.Linear(100, n_actions)
        self.out_value = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        policy = F.softmax(self.out_policy(x))
        value = self.out_value(x)
        return policy, value


def a3c_actor_wrapper(model):
    def _wrap(x):
        x = model(x)
        return x[0]
    return _wrap


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

    agent = ptan.agent.PolicyAgent(a3c_actor_wrapper(model))
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=run.getint("defaults", "n_steps"))

    optimizer = optim.RMSprop(model.parameters(), lr=run.getfloat("learning", "lr"))

    batch = []

    entropy_beta = run.getfloat("defaults", "entropy_beta", fallback=0.0)

    def calc_loss(batch):
        """
        Calculate loss from experience batch
        :param batch: list of experience entries
        :return: loss variable
        """
        # quite inefficient way, should be reordered to minimize amount of model() calls
        result = Variable(torch.FloatTensor(1).zero_())

        # extra values to monitor
        mon_adv = []
        mon_val_loss = []
        mon_pol_loss = []
        mon_ent_loss = []

        for exps in batch:
            v = Variable(torch.from_numpy(np.array([exps[0].state, exps[-1].state], dtype=np.float32)))
            policy_s, value_s = model(v)
            t_policy_s = policy_s[0]
            t_value_s = value_s[0]
            t_value_last_s = value_s[1]
            R = 0.0 if exps[-1].done else t_value_last_s.data.cpu().numpy()[0]
            for exp in reversed(exps):
                R *= GAMMA
                R += exp.reward
            advantage = R - t_value_s.data.cpu().numpy()[0]
            # policy loss part
            loss_policy = -t_policy_s.log()[exps[0].action] * advantage
            # value loss part
            loss_value = 0.5 * (t_value_s - R) ** 2
            # entropy loss
            loss_entropy = entropy_beta * (t_policy_s*t_policy_s.log()).sum()
            result += loss_policy + loss_value + loss_entropy

            # monitor stuff
            mon_adv.append(advantage)
            mon_val_loss.append(loss_value.data.cpu().numpy()[0])
            mon_pol_loss.append(loss_policy.data.cpu().numpy()[0])
            mon_ent_loss.append(loss_entropy.data.cpu().numpy()[0])

        monitor = {
            'advantage': np.mean(mon_adv),
            'value_loss': np.mean(mon_val_loss),
            'policy_loss': np.mean(mon_pol_loss),
            'entropy_loss': np.mean(mon_ent_loss),
        }

        return result / len(batch), monitor

    losses = []
    rewards = []
    iter_idx = 0

    graph_data = {
        'full_loss': [],
        'rewards': [],
        'advantage': [],
        'value_loss': [],
        'policy_loss': [],
        'entropy_loss': []
    }

    mean_games = run.getint("stop", "mean_games")
    mean_reward = run.getfloat("stop", "mean_reward")

    for exp in exp_source:
        batch.append(exp)
        if len(batch) < run.getint("learning", "batch_size"):
            continue

        # handle batch with experience
        optimizer.zero_grad()
        loss, monitor = calc_loss(batch)
        loss.backward()
        optimizer.step()
        f_loss = loss.data.cpu().numpy()[0]
        batch = []

        iter_idx += 1
        losses.append(f_loss)
        new_rewards = exp_source.pop_total_rewards()
        rewards.extend(new_rewards)
        losses = losses[-10:]
        rewards = rewards[-mean_games:]

        print("%d: mean_loss=%.3f, mean_reward=%.3f, done_games=%d, last_10_rewards=%s" % (
            iter_idx, 0.0 if not losses else np.mean(losses),
            0.0 if not rewards else np.mean(rewards), len(new_rewards),
            ", ".join(map(str, rewards[-10:]))
        ))
        graph_data['full_loss'].append(f_loss)
        graph_data['rewards'].extend(new_rewards)
        for k, v in monitor.items():
            graph_data[k].append(v)

        if rewards and np.mean(rewards) > mean_reward:
            break

    # plot charts
    output_file("a2c.html")

    figures = []
    f = figure(title="Full loss")
    f.line(range(iter_idx), graph_data['full_loss'])
    figures.append(f)

    f = figure(title="Policy loss")
    f.line(range(iter_idx), graph_data['policy_loss'])
    figures.append(f)

    f = figure(title="Value loss")
    f.line(range(iter_idx), graph_data['value_loss'])
    figures.append(f)

    f = figure(title="Entropy loss")
    f.line(range(iter_idx), graph_data['entropy_loss'])
    figures.append(f)

    f = figure(title="Rewards")
    f.line(range(iter_idx), graph_data['rewards'])
    figures.append(f)

    f = figure(title="Advantage")
    f.line(range(iter_idx), graph_data['advantage'])
    figures.append(f)

    show(bokeh.layouts.column(*figures))

#!/usr/bin/env python
import os
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

from ptan.common import runfile, env_params, utils, wrappers
from ptan.actions.epsilon_greedy import ActionSelectorEpsilonGreedy
from ptan import experience

GAMMA = 0.99

SAVE_INTERVAL = 20

class Net(nn.Module):
    def __init__(self, n_actions, input_shape=(1, 80, 80)):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 2)
#        self.conv4 = nn.Conv2d(64, 64, 2)

        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_size, 50)
        self.fc2 = nn.Linear(50, n_actions)

    def _get_conv_output(self, shape):
        input = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(input)
        print("Conv out shape: %s" % str(output_feat.size()))
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
#        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))
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
    parser.add_argument("-s", "--save", help="Directory to save model state")
    args = parser.parse_args()

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)

    run = runfile.RunFile(args.runfile)

    cuda_enabled = run.getboolean("defaults", "cuda", fallback=False)
    make_env = lambda: wrappers.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make(run.get("defaults", "env")))),
                                                width=80, height=80, grayscale=True)
    env = make_env()

    params = env_params.EnvParams.from_env(env)
    env_params.register(params)

    model = Net(params.n_actions)
    if cuda_enabled:
        model.cuda()

    loss_fn = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=run.getfloat("learning", "lr"))

    action_selector = ActionSelectorEpsilonGreedy(epsilon=run.getfloat("defaults", "epsilon"), params=params)

    def agent(states):
        """
        Return actions to take by a batch of states
        :param states: numpy array with states 
        :return: 
        """
        # TODO: move this into separate class
        v = Variable(torch.from_numpy(np.array(states, dtype=np.float32)))
        if cuda_enabled:
            v = v.cuda()
        q = model(v)
        actions = action_selector(q)
        return actions.data.cpu().numpy()

    exp_source = experience.ExperienceSource(env=env, agent=agent, steps_count=run.getint("defaults", "n_steps"))
    exp_replay = experience.ExperienceReplayBuffer(exp_source, buffer_size=run.getint("exp_buffer", "size"))

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

        if idx % SAVE_INTERVAL == 0 and idx > 0:
            if args.save:
                path = os.path.join(args.save, "model-%05d.dat" % idx)
                with open(path, 'wb') as fd:
                    torch.save(model.state_dict(), fd)
                print("Model %s saved" % path)

    env.close()
    pass

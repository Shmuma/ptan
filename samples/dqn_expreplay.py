import numpy as np

from ptan.common import env_params
from ptan.actions.epsilon_greedy import ActionSelectorEpsilonGreedy
from ptan import experience

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import gym

GAMMA = 0.99


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    params = env_params.EnvParams.from_env(env)
    env_params.register(params)

    model = nn.Sequential(
        nn.Linear(params.state_shape[0], 50),
        nn.ReLU(),
        nn.Linear(50, params.n_actions)
    )

    loss_fn = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    action_selector = ActionSelectorEpsilonGreedy(epsilon=0.05, params=params)

    def agent(states):
        """
        Return actions to take by a batch of states
        :param states: numpy array with states 
        :return: 
        """
        # TODO: move this into separate class
        v = Variable(torch.from_numpy(np.array(states, dtype=np.float32)))
        q = model(v)
        actions = action_selector(q)
        return actions.data.numpy()

    #
    # test_s = Variable(torch.from_numpy(np.array([env.reset()], dtype=np.float32)))
    # print(model(test_s))
    # print(action_selector(model(test_s)))
    # print(loss_fn(model(test_s), Variable(torch.Tensor([[1.0, 0.0, 2.0]]))))

    exp_source = experience.ExperienceSource(env=env, agent=agent, steps_count=1)
    exp_replay = experience.ExperienceReplayBuffer(exp_source, buffer_size=1000)

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
        return torch.from_numpy(np.array(states, dtype=np.float32)), torch.cat(q_vals)

    for idx in range(100000):
        exp_replay.populate(500)
        for batch in exp_replay.batches(50):
            optimizer.zero_grad()
            # populate buffer
            states, q_vals = batch_to_train(batch)
            # ready to train
            states, q_vals = Variable(states), Variable(q_vals)
            l = loss_fn(model(states), q_vals)
            l.backward()
            optimizer.step()
        if idx % 10 == 0:
            print("%d: Mean Q: %s" % (idx, q_vals.data.mean()))
    pass

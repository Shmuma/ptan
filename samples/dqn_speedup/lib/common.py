import sys
import time
import numpy as np
import torch
import torch.nn as nn

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from functools import reduce
import matplotlib.pylab as plt
import itertools
import csv
try:
    import telemetry
except:
    print('couldnt find telemetry package')

HYPERPARAMS = {
    'fsa-pong': {
        'env_name':         "fsa-PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    },
    'pong': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    },
    'breakout-small': {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    },
    'breakout': {
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    },
    'invaders': {
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'invaders',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
    'fsa-invaders': {
        'env_name': "fsa-SpaceInvadersNoFrameskip-v4",
        'stop_reward': 10.0,
        'run_name': 'fsa-invaders',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        # 'learning_rate': 0.00025,
        'learning_rate': 0.00005,
        'gamma': 0.99,
        'batch_size': 32,
        'video_interval': 1000000,
        'frame_stop': 1000001
    },
    'mr': {
        'env_name': "MontezumaRevengeNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'fsa-mr',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
    'fsa-mr': {
        'env_name': "fsa-MontezumaRevengeNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'fsa-mr',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
}


def unpack_batch(batch, fsa=False):
    if fsa:
        states, logics, actions, rewards, dones, last_states, last_logics = [], [], [], [], [], [], []
        for exp in batch:
            state = np.array(exp.state['image'], copy=False)
            states.append(state)
            logic = np.array(exp.state['logic'], copy=False)
            logics.append(logic)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(state)       # the result will be masked anyway
                last_logics.append(logic)
            else:
                last_states.append(np.array(exp.last_state['image'], copy=False))
                last_logics.append(np.array(exp.last_state['logic'], copy=False))
        return np.array(states, copy=False), np.array(logics, copy=False), np.array(actions), \
               np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), \
               np.array(last_states, copy=False), np.array(last_logics, copy=False) # , \
    else:
        states, actions, rewards, dones, last_states = [], [], [], [], []
        for exp in batch:
            state = np.array(exp.state, copy=False)
            states.append(state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(state)       # the result will be masked anyway
            else:
                last_states.append(np.array(exp.last_state, copy=False))
        return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, cuda=False, cuda_async=False, fsa=False, tm_net=None):
    if fsa:
        states, logics, actions, rewards, dones, next_states, next_logics = unpack_batch(batch, fsa)
        states_v = torch.tensor(states)
        logics_v = torch.tensor(logics)
        next_states_v = torch.tensor(next_states)
        next_logics_v = torch.tensor(next_logics)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        done_mask = torch.ByteTensor(dones)
        if cuda:
            states_v = states_v.cuda(non_blocking=cuda_async)
            logics_v = logics_v.cuda(non_blocking=cuda_async)
            next_states_v = next_states_v.cuda(non_blocking=cuda_async)
            next_logics_v = next_logics_v.cuda(non_blocking=cuda_async)
            actions_v = actions_v.cuda(non_blocking=cuda_async)
            rewards_v = rewards_v.cuda(non_blocking=cuda_async)
            done_mask = done_mask.cuda(non_blocking=cuda_async)

        if net.__class__.__name__ == 'FSADQNATTNMatching' or net.__class__.__name__ == 'FSADQNATTNMatchingFC':
            state_action_values, rr, cc = net({'image': states_v, 'logic': logics_v})
            state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            next_state_values, rr2, cc2 = tgt_net({'image': next_states_v, 'logic': next_logics_v})
        elif net.__class__.__name__ == 'FSADQNAppendToFCL1Conv':
            state_action_values, conv_out = net({'image': states_v, 'logic': logics_v})
            state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            next_state_values, conv_out2 = tgt_net({'image': next_states_v, 'logic': next_logics_v})
        else:
            state_action_values = net({'image': states_v, 'logic': logics_v})
            state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            next_state_values = tgt_net({'image': next_states_v, 'logic': next_logics_v})
        next_state_values = next_state_values.max(1)[0]
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        if net.__class__.__name__ == 'FSADQNATTNMatching' or net.__class__.__name__ == 'FSADQNATTNMatchingFC':
            dqn_loss = nn.MSELoss()(state_action_values, expected_state_action_values) + nn.MSELoss()(rr, cc.data)
        elif net.__class__.__name__ == 'FSADQNAppendToFCL1Conv':
            dqn_loss = nn.MSELoss()(state_action_values, expected_state_action_values) + 0.01*conv_out.norm(1)
        else:
            dqn_loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        if tm_net:
            predicted_logic = tm_net({'image': states_v, 'logic': logics_v}, actions_v)
            flat_next_logic = next_logics_v.view(next_logics_v.shape[0], -1).t()
            tm_loss = reduce(lambda x,y: x+y, map(lambda input, target: nn.CrossEntropyLoss()(input, target), predicted_logic, flat_next_logic))

        return dqn_loss, tm_loss

    else:
        states, actions, rewards, dones, next_states = unpack_batch(batch, fsa)
        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        done_mask = torch.ByteTensor(dones)
        if cuda:
            states_v = states_v.cuda(non_blocking=cuda_async)
            next_states_v = next_states_v.cuda(non_blocking=cuda_async)
            actions_v = actions_v.cuda(non_blocking=cuda_async)
            rewards_v = rewards_v.cuda(non_blocking=cuda_async)
            done_mask = done_mask.cuda(non_blocking=cuda_async)

        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)

class RewardTracker:
    def __init__(self, writer, stop_reward, telem=False, plot=False):
        self.writer = writer
        self.stop_reward = stop_reward
        self.rewards = np.array([])
        self.scores = np.array([])
        self.mean_scores = np.array([])
        self.count = 0
        if plot:
            f, (self.ax1, self.ax2) = plt.subplots(2, 1)
            plt.ion()
            plt.show()
        self.telemetry = telem
        if telem:
            self.tm = telemetry.ApplicationTelemetry()
        if telem:
            if not os.path.exists('/results'):
                os.makedirs('/results')
            self.outfile = '/results/output.txt'
        else:
            curdir = os.path.abspath(__file__)
            results = os.path.abspath(os.path.join(curdir, '../../../../results'))
            if not os.path.exists(results):
                os.makedirs(results)
            self.outfile = os.path.join(results, 'output.txt')

        self.fieldnames = ['frames', 'games', 'mean reward', 'mean score', 'max score']
        with open(self.outfile, 'w', newline='') as csvfile:

            csv_writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            csv_writer.writeheader()

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = [0]
        self.total_scores = [0]
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, score, frame, epsilon=None, plot=False):
        self.total_rewards.append(reward)
        if not score:
            self.total_scores.append(self.total_scores[-1])
        else:
            self.total_scores.append(score)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        max_score = np.max(self.total_scores)
        mean_score = np.mean(self.total_scores[-10:])
        if not epsilon:
            epsilon_str=  ""
        elif isinstance(epsilon, float):
            epsilon_str = ", eps %.2f" % epsilon
        elif isinstance(epsilon, dict):
            epsilon_str = ", eps %s" % str(epsilon)
        # epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        self.rewards = np.append(self.rewards, mean_reward)
        self.scores= np.append(self.scores, max_score)
        self.mean_scores = np.append(self.mean_scores, mean_score)
        self.count += 1
        if plot and self.count % 10 == 0:
            self.ax1.plot(self.rewards, 'r-')
            self.ax2.plot(self.scores, 'bo', self.mean_scores, 'b-')
            plt.pause(0.001)


        sys.stdout.flush()
        if epsilon is not None and not isinstance(epsilon, dict):
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)

        if self.telemetry:
            self.tm.metric_push_async({'metric': 'mean reward', 'value': mean_reward})
            self.tm.metric_push_async({'metric': 'mean score', 'value': mean_score})
            self.tm.metric_push_async({'metric': 'max score', 'value': max_score})

        with open(self.outfile, "a") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            csv_writer.writerow({'frames': frame, 'games': len(self.total_rewards),
                                 'mean reward': mean_reward, 'mean score': mean_score,
                                 'max score': max_score})
            # f.write("frames %d, games %d, mean reward %.3f, mean score %.3f, max score %.3f \n" % (
            # frame, len(self.total_rewards), mean_reward, mean_score, max_score
            # ))

        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
        # print(self.epsilon_greedy_selector.epsilon)

class IndexedEpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params, fsa_nvec):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']

        self.count_dict = {}
        all_fsa_states = map(lambda n: range(n), fsa_nvec)
        for element in itertools.product(*all_fsa_states):
            self.count_dict[element] = {'frame':0, 'count': 0}
            self.frame(element)

    def frame(self, logic):
        # right now 10 serves as the increment value for frame
        self.count_dict[logic]['count'] += 1
        frame = int(self.count_dict[logic]['count'] / 2)
        if frame > self.count_dict[logic]['frame']:
            self.epsilon_greedy_selector.epsilon_dict[logic] = \
                max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
            self.count_dict[logic]['frame'] = frame
            # print(logic, frame, self.epsilon_greedy_selector.epsilon_dict[logic])

class IndexedEpsilonTrackerNoStates:
    def __init__(self, epsilon_greedy_selector, params, fsa_nvec):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']

        self.count = 0
        self.frame_count = 0
        all_fsa_states = map(lambda n: range(n), fsa_nvec)
        for element in itertools.product(*all_fsa_states):
            self.frame(element)

    def frame(self, logic):
        # right now 10 serves as the increment value for frame
        self.count += 1
        frame = int(self.count )
        if frame > self.frame_count:
            new_eps = max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
            for key in self.epsilon_greedy_selector.epsilon_dict:
                self.epsilon_greedy_selector.epsilon_dict[key] = new_eps
            self.frame_count = frame
            # print(logic, frame, self.epsilon_greedy_selector.epsilon_dict[logic])
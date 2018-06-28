import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import itertools

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
        'stop_reward': 500.0,
        'run_name': 'fsa-invaders',
        'replay_size': 10 ** 6,
        # 'replay_initial': 10000,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        # 'learning_rate': 0.00005,
        'gamma': 0.99,
        'batch_size': 32
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
        # states, logics, actions, rewards, dones, last_states, last_logics, \
        #     recons, conv_outs = [], [], [], [], [], [], [], [], []
        states, logics, actions, rewards, dones, last_states, last_logics = [], [], [], [], [], [], []
        for exp in batch:
            state = np.array(exp.state['image'], copy=False)
            states.append(state)
            logic = np.array(exp.state['logic'], copy=False)
            logics.append(logic)
            # recon = np.array(exp.recon.data, copy=False)
            # recons.append(recon)
            # conv_out = np.array(exp.conv_out.data, copy=False)
            # conv_outs.append(conv_out)
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
               # np.array(recons, copy=False), np.array(conv_outs, copy=False)
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


def calc_loss_dqn(batch, net, tgt_net, gamma, cuda=False, cuda_async=False, fsa=False):


    if fsa:
        # states, logics, actions, rewards, dones, next_states, next_logics, \
        #     recons, conv_outs = unpack_batch(batch, fsa)
        states, logics, actions, rewards, dones, next_states, next_logics = unpack_batch(batch, fsa)
        states_v = torch.tensor(states)
        logics_v = torch.tensor(logics)
        next_states_v = torch.tensor(next_states)
        next_logics_v = torch.tensor(next_logics)
        # recons_v = torch.tensor(recons)
        # conv_outs_v = torch.tensor(conv_outs)
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
            # recons_v = recons_v.cuda(non_blocking=cuda_async)
            # conv_outs_v = conv_outs_v.cuda(non_blocking=cuda_async)

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
            return nn.MSELoss()(state_action_values, expected_state_action_values) + nn.MSELoss()(rr, cc.data)
        elif net.__class__.__name__ == 'FSADQNAppendToFCL1Conv':
            return nn.MSELoss()(state_action_values, expected_state_action_values) + 0.01*conv_out.norm(1)
        else:
            return nn.MSELoss()(state_action_values, expected_state_action_values)

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
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward
        self.rewards = np.array([])
        self.count = 0
        plt.axis()
        plt.ion()
        plt.show()

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
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
        self.count += 1
        if self.count % 10 == 0:
            plt.plot(self.rewards, '-')
            plt.draw()
            plt.pause(0.001)


        sys.stdout.flush()
        if epsilon is not None and not isinstance(epsilon, dict):
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
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
        frame = int(self.count / 2)
        if frame > self.frame_count:
            new_eps = max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
            for key in self.epsilon_greedy_selector.epsilon_dict:
                self.epsilon_greedy_selector.epsilon_dict[key] = new_eps
            self.frame_count = frame
            # print(logic, frame, self.epsilon_greedy_selector.epsilon_dict[logic])
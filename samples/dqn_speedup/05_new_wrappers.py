#!/usr/bin/env python3
import ptan
import argparse

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from lib import dqn_model, common, atari_wrappers
import json

PLAY_STEPS = 4


def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'], fsa=params['fsa'])
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True, pytorch_img=True, fsa=params['fsa'])
    return env


def play_func(params, net, cuda, fsa, exp_queue, fsa_nvec=None):
    device = torch.device("cuda" if cuda else "cpu")
    env = make_env(params)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-05_new_wrappers")
    if not fsa:
        selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
        epsilon_tracker = common.EpsilonTracker(selector, params)
        agent = ptan.agent.DQNAgent(net, selector, device=device, fsa=fsa)
    else:
        selector = ptan.actions.EpsilonGreedyActionSelectorFsa(fsa_nvec, epsilon=params['epsilon_start'])
        if 'Index' in net.__class__.__name__:
            epsilon_tracker = common.IndexedEpsilonTracker(selector, params, fsa_nvec)
        else:
            epsilon_tracker = common.IndexedEpsilonTrackerNoStates(selector, params, fsa_nvec)
        agent = ptan.agent.DQNAgent(net, selector, device=device, fsa=fsa, epsilon_tracker=epsilon_tracker)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    exp_source_iter = iter(exp_source)

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward'], params['telemetry']) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            if not fsa:
                epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            new_scores = exp_source.pop_total_scores()
            if new_rewards:
                if not fsa:
                    new_score = [] if not new_scores else new_scores[0]
                    if reward_tracker.reward(new_rewards[0], new_score, frame_idx, selector.epsilon, params['plot']):
                        break
                else:
                    new_score = [] if not new_scores else new_scores[0]
                    if reward_tracker.reward(new_rewards[0], new_score, frame_idx, selector.epsilon_dict, params['plot']):
                        break

    exp_queue.put(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--fsa", default=False, action="store_true", help="Use FSA stuff")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot reward")
    parser.add_argument("--telemetry", default=False, action="store_true", help="Use telemetry")
    parser.add_argument("--file", default='', help="Config file")

    args = parser.parse_args()

    mp.set_start_method('spawn')
    if args.fsa:
        params = common.HYPERPARAMS['fsa-invaders']
    else:
        params = common.HYPERPARAMS['invaders']
    params['batch_size'] *= PLAY_STEPS
    params['fsa'] = args.fsa
    params['plot'] = args.plot
    params['telemetry'] = args.telemetry

    if args.file:
        config_file = json.loads(open(args.file, "r").read())
        for option in params:
            if option in config_file:
                params[option] = config_file[option]

    device = torch.device("cuda" if args.cuda else "cpu")

    env = make_env(params)

    if args.fsa:
        net = dqn_model.FSADQNParallel(env.observation_space.spaces['image'].shape,
                                           env.observation_space.spaces['logic'].nvec,
                                           env.action_space.n).to(device)
        # net = dqn_model.FSADQNConvOneLogic(env.observation_space.spaces['image'].shape,
        #                        env.observation_space.spaces['logic'].nvec, env.action_space.n).to(device)
    else:
        net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    # optimizer = optim.RMSprop(net.parameters(), lr=params['learning_rate'], momentum=0.95, eps=0.01)

    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)

    if args.fsa:
        fsa_nvec = env.observation_space.spaces['logic'].nvec
        logic_dim = int(fsa_nvec.shape[0] / env.observation_space.spaces['image'].shape[0])
        fsa_nvec = fsa_nvec[-logic_dim:]
        play_proc = mp.Process(target=play_func,
                               args=(params, net, args.cuda, args.fsa, exp_queue,
                                     fsa_nvec))
    else:
        play_proc = mp.Process(target=play_func, args=(params, net, args.cuda, args.fsa, exp_queue))

    play_proc.start()

    frame_idx = 0

    while play_proc.is_alive():
        # build up experience replay buffer?
        frame_idx += PLAY_STEPS
        for _ in range(PLAY_STEPS):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params['replay_initial']:
            continue

        # train on ERB?
        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'],
                                      cuda=args.cuda, cuda_async=True, fsa=args.fsa)
        loss_v.backward()
        optimizer.step()

        if frame_idx % params['target_net_sync'] < PLAY_STEPS:
            tgt_net.sync()

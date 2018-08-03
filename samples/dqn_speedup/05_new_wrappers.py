#!/usr/bin/env python3
import ptan
import argparse

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from lib import dqn_model, common, atari_wrappers
import json
import os
import pickle
import numpy as np

from gym import wrappers

import distutils.spawn, distutils.version

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

    with common.RewardTracker(writer, params['stop_reward'], params['telemetry'], params['plot']) as reward_tracker:
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
    parser.add_argument("--video", default=False, action="store_true", help="Record video")
    parser.add_argument("--telemetry", default=False, action="store_true", help="Use telemetry")
    parser.add_argument("--file", default='', help="Config file")

    args = parser.parse_args()

    mp.set_start_method('spawn')

    model = None

    if args.file:
        config_file = json.loads(open(args.file, "r").read())
        if 'dqn_model' in config_file:
            model = config_file['dqn_model']
        if 'fsa' in config_file:
            args.fsa = config_file['fsa']

    if args.fsa:
        params = common.HYPERPARAMS['fsa-invaders']
    else:
        params = common.HYPERPARAMS['invaders']
    params['batch_size'] *= PLAY_STEPS
    params['fsa'] = args.fsa
    params['plot'] = args.plot
    params['telemetry'] = args.telemetry

    if args.file:
        for option in params:
            if option in config_file:
                params[option] = config_file[option]

    device = torch.device("cuda" if args.cuda else "cpu")

    curdir = os.path.abspath(__file__)
    if args.telemetry:
        model_path = '/results/model'
        params_path = '/results'
    else:
        model_path = os.path.abspath(os.path.join(curdir, '../../../results/model'))
        params_path = os.path.abspath(os.path.join(curdir, '../../../results'))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if args.video:
        if args.telemetry:
            video_path = '/results/video'
        else:
            video_path = os.path.abspath(os.path.join(curdir, '../../../results/video'))

        if not os.path.exists(video_path):
            os.makedirs(video_path)

    with open(params_path + "/params.txt", "w+") as f:
        for param, value in params.items():
            f.write(param + ": " + str(value) + "\n")

    if distutils.spawn.find_executable('avconv') is not None:
        print('using avconv')
    elif distutils.spawn.find_executable('ffmpeg') is not None:
        print('using ffmpeg')
    else:
        print('has neither avconv nor ffmpeg')

    env = make_env(params)

    if args.fsa:
        net = dqn_model.FSADQNBiasIndex(env.observation_space.spaces['image'].shape,
                                           env.observation_space.spaces['logic'].nvec,
                                           env.action_space.n).to(device)
        model_name = 'FSADQNBiasIndex'
        # net = dqn_model.FSADQNConvOneLogic(env.observation_space.spaces['image'].shape,
        #                        env.observation_space.spaces['logic'].nvec, env.action_space.n).to(device)
    else:
        net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
        model_name = 'DQN'

    dqn_models = {
        'FSADQN': dqn_model.FSADQN,
        'FSADQNBias': dqn_model.FSADQNBias,
        'FSADQNBiasIndex': dqn_model.FSADQNBiasIndex,
        'FSADQNScaling': dqn_model.FSADQNScaling,
        'FSADQNAffine': dqn_model.FSADQNAffine,
        'FSADQNIndexOutput' : dqn_model.FSADQNIndexOutput,
        'FSADQNATTNMatchingFC': dqn_model.FSADQNATTNMatchingFC,
        'FSADQNATTNMatching': dqn_model.FSADQNATTNMatching,
        'FSADQNAppendToFC': dqn_model.FSADQNAppendToFC,
        'FSADQNAppendToFCL1Conv': dqn_model.FSADQNAppendToFCL1Conv,
        'FSADQNIndexConv': dqn_model.FSADQNIndexConv,
        'FSADQNIndexConvOneLogic': dqn_model.FSADQNIndexConvOneLogic,
        'FSADQNIndexATTN': dqn_model.FSADQNIndexATTN
    }

    if model and model in dqn_models:
        net = dqn_models[model](env.observation_space.spaces['image'].shape,
                                       env.observation_space.spaces['logic'].nvec,
                                       env.action_space.n).to(device)
        model_name = model

    print("using model {}".format(model_name))

    tgt_net = ptan.agent.TargetNet(net)

    tm_net = dqn_model.TMPredict(env.observation_space.spaces['image'].shape,
                                       env.observation_space.spaces['logic'].nvec,
                                       env.action_space.n).to(device)

    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    optimizer_tm = optim.Adam(tm_net.parameters(), lr=params['learning_rate'])
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

    counter = 0
    while play_proc.is_alive():
        if frame_idx > params["frame_stop"]:
            play_proc.terminate()
            break
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
        optimizer_tm.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v, tm_loss = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'],
                                      cuda=args.cuda, cuda_async=True, fsa=args.fsa, tm_net=tm_net)
        loss_v.backward()
        optimizer.step()

        tm_loss.backward()
        optimizer_tm.step()

        if frame_idx > counter*params['video_interval'] and args.video:
            test_env = wrappers.Monitor(make_env(params),
                                        "{}/frame{}".format(video_path, counter),
                                        video_callable=lambda ep_id: True if ep_id < 3 else False,
                                        force=True)
            obs = test_env.reset()
            test_agent = ptan.agent.PolicyAgent(net, action_selector=ptan.actions.ArgmaxActionSelector(),
                                                device=device, fsa=args.fsa)
            real_done = False
            while not real_done:
                if args.plot:
                    test_env.render()
                actions, agent_states = test_agent([obs])
                obs, reward, done, info = test_env.step(actions[0])
                real_done = test_env.env.env.env.env.env.env.was_real_done
                if real_done:
                    print(test_env.env.env.env.env.env.env.score)
                if done:
                    obs = test_env.reset()

            model_video_path = "{}/frame{}/model.pth".format(video_path, counter)
            torch.save(net.state_dict(), model_video_path)
            test_env.close()
            counter += 1

        if frame_idx % params['target_net_sync'] < PLAY_STEPS:
            tgt_net.sync()

    eval_runs = 10
    eval_env = make_env(params)
    obs = eval_env.reset()
    eval_agent = ptan.agent.PolicyAgent(net, action_selector=ptan.actions.ArgmaxActionSelector(),
                                        device=device, fsa=args.fsa)
    score = 0
    correct_logic = 0
    total_logic = 0
    for i in range(eval_runs):
        next_logic_state = None
        real_done = False
        while not real_done:
            actions, agent_states = eval_agent([obs])
            obs, reward, done, info = eval_env.step(actions[0])
            if next_logic_state:
                if np.array_equal(np.array(next_logic_state), obs['logic'].reshape(-1)):
                    correct_logic += 1
                total_logic += 1
            next_logic_state = [int(torch.argmax(x).cpu()) for x in
                                    tm_net({'image': torch.from_numpy(obs['image']).cuda()[None, :],
                                            'logic': torch.from_numpy(obs['logic']).cuda()[None, :]},
                                           torch.from_numpy(actions).cuda())]

            real_done = eval_env.env.env.env.env.env.was_real_done
            if real_done:
                score += eval_env.env.env.env.env.env.score
                print("test: {} | score: {}".format(i, eval_env.env.env.env.env.env.score))
            if done:
                obs = eval_env.reset()
    eval_env.close()
    score /= eval_runs
    logic_acc = correct_logic/total_logic
    print("TM acc: {}".format(logic_acc))

    model_file = "/model.pth"
    torch.save(net.state_dict(), model_path + model_file)

    data_file = "/data.pkl"
    save_data = [score, logic_acc, model_name, params, args]

    with open(model_path + data_file, 'wb') as output:
        pickle.dump(save_data, output, pickle.HIGHEST_PROTOCOL)
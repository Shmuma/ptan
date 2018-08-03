#!/usr/bin/env python3
import ptan
import argparse

import torch
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from lib import dqn_model, common, atari_wrappers
import json
import os

import pickle

from gym import wrappers

def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'], fsa=params['fsa'])
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True, pytorch_img=True, fsa=params['fsa'])
    return env

if __name__ == "__main__":
    curdir = os.path.abspath(__file__)
    model_path = os.path.abspath(os.path.join(curdir, '../../../results/model'))
    data_file = "/data.pkl"
    with open(model_path + data_file, 'rb') as input:
        data = pickle.load(input)
        score = data[0]
        model_name = data[1]
        params = data[2]
        args = data[3]


    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", default=False, action="store_true", help="Plot reward")

    new_args = parser.parse_args()
    args.plot = new_args.plot

    device = torch.device("cuda" if args.cuda else "cpu")

    '''
    LOAD THE MODEL
    '''
    env = make_env(params)

    model_file = "/model.pth"
    model_state_dict = torch.load(model_path + model_file)

    dqn_models = {
        'FSADQN': dqn_model.FSADQN,
        'FSADQNParallel': dqn_model.FSADQNParallel,
        'FSADQNIndexOutput' : dqn_model.FSADQNIndexOutput,
        'FSADQNATTNMatchingFC': dqn_model.FSADQNATTNMatchingFC,
        'FSADQNATTNMatching': dqn_model.FSADQNATTNMatching,
        'FSADQNAppendToFC': dqn_model.FSADQNAppendToFC,
        'FSADQNAppendToFCL1Conv': dqn_model.FSADQNAppendToFCL1Conv,
        'FSADQNIndexConv': dqn_model.FSADQNIndexConv,
        'FSADQNIndexConvOneLogic': dqn_model.FSADQNIndexConvOneLogic,
        'FSADQNIndexATTN': dqn_model.FSADQNIndexATTN
    }

    net = dqn_models[model_name](env.observation_space.spaces['image'].shape,
                                   env.observation_space.spaces['logic'].nvec,
                                   env.action_space.n).to(device)

    net.load_state_dict(model_state_dict)

    obs = env.reset()
    agent = ptan.agent.PolicyAgent(net, action_selector=ptan.actions.ArgmaxActionSelector(),
                                        device=device, fsa=args.fsa)
    real_done = False
    while not real_done:
        if args.plot:
            env.render()
        actions, agent_states = agent([obs])
        obs, reward, done, info = env.step(actions[0])
        real_done = env.env.env.env.env.env.was_real_done
        if real_done:
            ram = env.unwrapped.ale.getRAM()
            aliens = ram[17]
            score = env.env.env.env.env.env.score
            print("score: {} | aliens: {}".format(score, aliens))
        if done:
            obs = env.reset()
    env.close()
import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools

import numpy as np

def constant_init_to_1(tensor):
    return nn.init.constant_(tensor, 1.0)

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)

class FSADQN(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQN, self).__init__()

        # input_shape[0] = number of input channels
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv_out_size = self._get_conv_out(input_shape)

        self.attention_map = dict()
        self.attention_layers = []
        self.fsa_final_layers = []
        all_fsa_states = map(lambda n: range(n), fsa_nvec)
        count = 0
        for element in itertools.product(*all_fsa_states):
            #lin_i = nn.Linear(in_features=self.conv_out_size,
            #              out_features=self.conv_out_size,
            #              bias=False)
            #lin_i.weight.data.fill_(1)
            #lv_i = nn.Sequential()
            #lv_i.add_module('linear', lin_i)
            #lv_i.add_module('softmax', nn.Softmax())

            # attention map depending on the input
            conv_i = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=self.conv_out_size, out_features=self.conv_out_size, bias=True),
            nn.Softmax()
            )

            # record dictionary
            e = tuple(element)
            if e not in self.attention_map:
                self.attention_map[e] = count
                count += 1
            self.attention_layers.append(conv_i)

            fsa_linear_i = nn.Linear(512, n_actions)
            self.fsa_final_layers.append(fsa_linear_i)

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU()
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))


    def forward(self, x):
        im = x['image']
        # note that we are only looking at the logic vector of
        # the 0th frame right now. should check that we look
        # at the vector of the most recent frame (may be last frame?)
        logic = tuple(x['logic'][0].view(-1).cpu().numpy()) # assume is a vector of the same shape as fsa_nvec
        fx = im.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        # make the mask
        #fsa_mask = torch.ones(self.conv_out_size, requires_grad=True).to('cuda')
        #mask_weights = self.attention_layers[self.attention_map[logic]].cuda()
        #fsa_mask = mask_weights(fsa_mask.view(-1)).view(self.conv_out_size)
        mask_weights = self.attention_layers[self.attention_map[logic]].cuda()
        mask_output = mask_weights(fx).view(fx.size()[0], -1)
        fsa_mask = mask_output # .view(self.conv_out_size)
        fc_output = self.fc(conv_out*fsa_mask.to('cuda'))
        fsa_final = self.fsa_final_layers[self.attention_map[logic]].cuda()
        return fsa_final(fc_output)

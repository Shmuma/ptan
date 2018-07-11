import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools
from functools import reduce

import numpy as np

def constant_init_to_1(tensor):
    return nn.init.constant_(tensor, 1.0)


def make_one_hot_logic(labels, fsa_nvec):
    # apply make_one_hot to every (logic_state, logic_dim) pair
    map_one_hot = map(lambda x, y: make_one_hot(x[None, :].t().type(torch.cuda.LongTensor), y), labels.t(), fsa_nvec)
    # append the dimensions together
    one_hot_logic = torch.cat(tuple(map_one_hot), 1)
    return one_hot_logic

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.cuda.LongTensor
        N x 1 x F, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.cuda.FloatTensor
        N x C x F, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

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

# control class. it's the same as the DQN model but can handle the logic
class FSADQN(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQN, self).__init__()

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
        x = x['image']
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

class TMPredict(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(TMPredict, self).__init__()

        self.fsa_nvec = torch.tensor(fsa_nvec).float().cuda()
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.conv_output = 32
        self.conv_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.conv_output)
        )

        self.out_fc = nn.Sequential(
            nn.Linear(self.conv_output + n_actions + int(fsa_nvec.sum()), 512),
            nn.ReLU(),
            nn.Linear(512, int(fsa_nvec.sum()))
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, a):
        image = x['image']
        fx = image.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        image_vec = self.conv_fc(conv_out)

        logic = x['logic'].view(fx.size()[0], -1).float()
        oh_logic = make_one_hot_logic(logic, self.fsa_nvec)
        oh_action = make_one_hot(a[None, :].t(), self.n_actions)

        cat_vec = torch.cat((image_vec, oh_logic, oh_action), 1)
        fc_out = self.out_fc(cat_vec)

        output_by_logic_dim = []
        start = 0
        for i in self.fsa_nvec.cpu().numpy().astype(int).tolist():
            logic_out = fc_out[:, start:start+i]
            start += i
            output_by_logic_dim.append(logic_out)

        return output_by_logic_dim

# estimate q values using image and fsa state in parallel
# and then add them up in the end
class FSADQNParallel(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNParallel, self).__init__()

        self.fsa_nvec = torch.tensor(fsa_nvec).float().cuda()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fsa_fc = nn.Sequential(
            nn.Linear(fsa_nvec.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
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
        image = x['image']
        fx = image.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        image_q = self.fc(conv_out)

        logic = x['logic'].view(fx.size()[0], -1).float()
        logic_q = self.fsa_fc(logic/self.fsa_nvec)
        return image_q + logic_q

# estimate q values using image and fsa state in parallel
# and then add them up in the end
class FSADQNScaling(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNScaling, self).__init__()

        self.fsa_nvec = torch.tensor(fsa_nvec).float().cuda()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fsa_fc = nn.Sequential(
            nn.Linear(fsa_nvec.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
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
        image = x['image']
        fx = image.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        image_q = self.fc(conv_out)

        logic = x['logic'].view(fx.size()[0], -1).float()
        logic_q = self.fsa_fc(logic/self.fsa_nvec)
        return image_q * logic_q

# estimate q values using image and fsa state in parallel
# and then add them up in the end
class FSADQNAffine(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNAffine, self).__init__()

        self.fsa_nvec = torch.tensor(fsa_nvec).float().cuda()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fsa_fc = nn.Sequential(
            nn.Linear(fsa_nvec.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.fsa_fc2 = nn.Sequential(
            nn.Linear(fsa_nvec.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
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
        image = x['image']
        fx = image.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        image_q = self.fc(conv_out)

        logic = x['logic'].view(fx.size()[0], -1).float()
        logic_q = self.fsa_fc(logic/self.fsa_nvec)
        logic_q2 = self.fsa_fc2(logic/self.fsa_nvec)
        return image_q * logic_q + logic_q2

# outputs a tensor that can be indexed by logic state
class FSADQNIndexOutput(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNIndexOutput, self).__init__()

        self.n_actions = n_actions

        fsa_dim = int(fsa_nvec.shape[0] / input_shape[0])
        self.num_fsa_states = int(np.prod(fsa_nvec[-fsa_dim:]))
        self.fsa_nvec = fsa_nvec[-fsa_dim:]

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
            nn.Linear(512, n_actions*self.num_fsa_states)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        image = x['image']
        logic = x['logic'][:, -1]
        fx = image.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        fc_out = self.fc(conv_out).view(image.shape[0], *self.fsa_nvec, self.n_actions)
        factored = fc_out[:, logic[:, 0], logic[:, 1]][:, 0]
        return factored


# implementation of Kiran's idea to use FSA state to predict conv output
# then get attention layer from the diff between predicted and actual
# conv output
# using FC layers instead of deconv
class FSADQNATTNMatchingFC(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNATTNMatchingFC, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        fsa_dim = int(fsa_nvec.shape[0]/input_shape[0])
        # output size determined by kernel_size
        self.deconv = nn.Sequential(
            nn.Linear(fsa_dim, 32),
            nn.Linear(32, conv_out_size)
        )

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2*n_actions),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(2*n_actions + fsa_dim, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        image = x['image']
        logic = x['logic'][:, -1]
        logic = logic.float()
        recon = self.deconv(logic)
        fx = image.float() / 256
        conv_out = self.conv(fx)
        conv_out = conv_out.view(conv_out.shape[0], -1)
        diff = (conv_out - recon).abs()
        max_diff = torch.max(diff)
        diff = diff / max_diff
        conv_out_masked = conv_out * diff
        fc_out = self.fc(conv_out_masked)
        appended_fsa = torch.cat((fc_out.t(), x['logic'][:, -1].float().t())).t()
        return self.fc2(appended_fsa), recon, conv_out


# implementation of Kiran's idea to use FSA state to predict conv output
# then get attention layer from the diff between predicted and actual
# conv output
class FSADQNATTNMatching(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNATTNMatching, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        fsa_dim = int(fsa_nvec.shape[0]/input_shape[0])
        # output size determined by kernel_size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(fsa_dim, 32, kernel_size=3, stride=1),
            nn.ConvTranspose2d(32, 64, kernel_size=5, stride=1)
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2*n_actions),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(2*n_actions + fsa_dim, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        image = x['image']
        logic = x['logic'][:, -1]
        logic = logic[..., None, None].float()
        recon = self.deconv(logic)
        fx = image.float() / 256
        conv_out = self.conv(fx)
        diff = (conv_out - recon).abs()
        diff_sm = nn.Softmax(dim=2)(diff.view(diff.shape[0], diff.shape[1], diff.shape[2]*diff.shape[3]))
        diff = diff_sm.view(diff.shape[0], diff.shape[1], diff.shape[2], diff.shape[3])
        # max_diff = torch.max(diff)
        # diff = diff / max_diff
        conv_out_masked = conv_out * diff
        conv_out_flat = conv_out_masked.view(fx.size()[0], -1)
        fc_out = self.fc(conv_out_flat)
        appended_fsa = torch.cat((fc_out.t(), x['logic'][:, -1].float().t())).t()
        return self.fc2(appended_fsa), recon, conv_out


# simply append FSA state to the FC layer
class FSADQNAppendToFC(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNAppendToFC, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        fsa_dims = fsa_nvec.shape[0]/input_shape[0]
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + fsa_dims, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        image = x['image']
        fx = image.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        appended_fsa = torch.cat((conv_out.t(), x['logic'][:, -1].float().t())).t()
        return self.fc(appended_fsa)

# simply append FSA state to the FC layer
# and put an L1 loss on the convolution...
class FSADQNAppendToFCL1Conv(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNAppendToFCL1Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        fsa_dims = fsa_nvec.shape[0]/input_shape[0]
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + fsa_dims, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        image = x['image']
        logic = x['logic'][:, -1]
        fx = image.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        appended_fsa = torch.cat((conv_out.t(), logic.float().t())).t()
        return self.fc(appended_fsa), conv_out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)

# shared conv net, no attention
class FSADQNIndexConv(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNIndexConv, self).__init__()

        # input_shape[0] = number of input channels
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.fsa_map = dict()
        self.final_conv_layers = []
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
            # final_conv layer depending on the input
            conv_i = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )

            # record dictionary
            e = tuple(element)
            if e not in self.fsa_map:
                self.fsa_map[e] = count
                count += 1
            self.final_conv_layers.append(conv_i)

            if count == 1:
                self.conv_out_size = self._get_conv_out(input_shape)

            fsa_linear_i = nn.Linear(512, n_actions)
            self.fsa_final_layers.append(fsa_linear_i)

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, n_actions),
            nn.ReLU()
        )

    def _get_conv_out(self, shape):
        o = self.final_conv_layers[0](self.conv(Variable(torch.zeros(1, *shape))))
        return int(np.prod(o.size()))


    def forward(self, x):
        im = x['image']
        # note that we are only looking at the logic vector of
        # the 0th frame right now. should check that we look
        # at the vector of the most recent frame (may be last frame?)
        logic = tuple(x['logic'][0].view(-1).cpu().numpy()) # assume is a vector of the same shape as fsa_nvec
        fx = im.float() / 256
        # select based on fsa state
        final_conv_layer = self.final_conv_layers[self.fsa_map[logic]].cuda()
        # compose with shared conv net
        conv_out = final_conv_layer(self.conv(fx)).view(fx.size()[0], -1)
        fc_output = self.fc(conv_out)
        # select based on fsa state
        # fsa_final = self.fsa_final_layers[self.fsa_map[logic]].cuda()
        # return fsa_final(fc_output)
        return fc_output

# shared conv net, no attention
# this version has been modified to use only the LAST set of logic states
# from the 4 frames
class FSADQNIndexConvOneLogic(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNIndexConvOneLogic, self).__init__()

        # input_shape[0] = number of input channels
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.logic_dim = int(fsa_nvec.shape[0]/input_shape[0])
        fsa_nvec = fsa_nvec[-self.logic_dim:]

        self.fsa_map = dict()
        self.final_conv_layers = []
        self.fsa_final_layers = []
        all_fsa_states = map(lambda n: range(n), fsa_nvec)
        count = 0
        for element in itertools.product(*all_fsa_states):

            conv_i = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )

            # record dictionary
            e = tuple(element)
            if e not in self.fsa_map:
                self.fsa_map[e] = count
                count += 1
            self.final_conv_layers.append(conv_i)

            if count == 1:
                self.conv_out_size = self._get_conv_out(input_shape)

            fsa_linear_i = nn.Linear(512, n_actions)
            self.fsa_final_layers.append(fsa_linear_i)

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.final_conv_layers[0](self.conv(Variable(torch.zeros(1, *shape))))
        return int(np.prod(o.size()))


    def forward(self, x):
        im = x['image']
        logic = tuple(x['logic'][0][-1].view(-1).cpu().numpy()) # assume is a vector of the same shape as fsa_nvec
        fx = im.float() / 256
        final_conv_layer = self.final_conv_layers[self.fsa_map[logic]].cuda()
        conv_out = final_conv_layer(self.conv(fx)).view(fx.size()[0], -1)
        fc_output = self.fc(conv_out)

        return fc_output

# attentional version
class FSADQNIndexATTN(nn.Module):
    def __init__(self, input_shape, fsa_nvec, n_actions):
        super(FSADQNIndexATTN, self).__init__()

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

        self.fsa_map = dict()
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
            # should make this smaller (have to calculate proper size then)
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
            if e not in self.fsa_map:
                self.fsa_map[e] = count
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
        mask_weights = self.attention_layers[self.fsa_map[logic]].cuda()
        mask_output = mask_weights(fx).view(fx.size()[0], -1)
        fsa_mask = mask_output # .view(self.conv_out_size)
        fc_output = self.fc(conv_out*fsa_mask.to('cuda'))
        fsa_final = self.fsa_final_layers[self.fsa_map[logic]].cuda()
        return fsa_final(fc_output)

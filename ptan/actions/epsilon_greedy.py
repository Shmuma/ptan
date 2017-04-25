import numpy as np

import torch.nn as nn

from ..common import env_params


class SelectAction(nn.Module):
    def __init__(self, epsilon=0.05, params=env_params.get()):
        super(SelectAction, self).__init__()
        self.epsilon = epsilon
        self.params = params

    def forward(self, q_vals):
        batch_size = q_vals.size()[0]
        res = q_vals.max(dim=1)[1]
        for i in range(batch_size):
            if np.random.rand() < self.epsilon:
                res[i] = np.random.choice(params.n_actions)
        return res

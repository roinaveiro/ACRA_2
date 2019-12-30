import torch
import torch.nn as nn
import torch.nn.functional as F


# check what to do with the biases of the linear layers. How do they do in Gal et al?
class MLP(nn.Module):
    """ A standard feedforward neural network with options for residual connections and dropouts """

    def __init__(self, in_dim, out_dim, hid_dim, other_args):
        super(MLP, self).__init__()
        self.d_in = in_dim
        #self.dropout = dropout

        self.n_layers = other_args.n_layers
        self.fc_in = nn.Linear(self.d_in, hid_dim)
        self.fc_inners = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim, bias=False) for _ in range(self.n_layers)])
        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.args = other_args

    def forward(self, x):
        x = x.reshape(x.shape[0], self.d_in)
        x = F.relu(self.fc_in(x))

        for l, fc_inner in enumerate(self.fc_inners):

            #x_d = self.dropout(x, l)
            x = F.relu(fc_inner(x))

        x = self.fc_out(x)
        return x


# class Dropout(nn.Module):
#     """ This module adds (standard Bernoulli) Dropout to the following weights of a layer.
#     """
#
#     def __init__(self, p=0.1):
#         super(Dropout, self).__init__()
#         assert p <= 1.
#         assert p >= 0.
#         self.p = p
#
#     def forward(self, x, context=None):
#         if self.training:
#             binomial = torch.distributions.binomial.Binomial(
#                 probs=torch.tensor(1-self.p, device=x.device))
#             x = x * binomial.sample(x.size()) * \
#                 (1. / (1. - self.p))   # inverted dropout
#         return x

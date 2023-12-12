import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from PyIF import te_compute as te


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.node_weights = {}
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # for i in range(input.size(0)):
        #     for j in range(input.size(0)):
        #         if i != j:
        #             # Assuming input is a feature matrix with nodes as rows
        #             node_i_features = input[i]
        #             node_j_features = input[j]
        #
        #             # Calculate transfer entropy from node i to node j
        #             #te_value = te.te_compute(node_i_features, node_j_features, k=1, embedding=1, safetyCheck=False, GPU=False)
        #

        #te.te_compute(X_1000, Y_1000, k=1, embedding=1, safetyCheck=True, GPU=False)

        #te1 = te.te_compute(adj.to_dense().cpu().numpy()[222].flatten(), adj.to_dense().cpu().numpy()[1].flatten(), k=1, embedding=1, safetyCheck=True, GPU=False)
        #te1 = te.te_compute(adj.to_dense().cpu().numpy()[222].flatten(), adj.to_dense().cpu().numpy()[1].flatten(), k=100, embedding=1, safetyCheck=True, GPU=False)
        #te1 = te.te_compute(adj.to_dense().cpu().numpy()[222].flatten(), adj.to_dense().cpu().numpy()[1].flatten(), k=1, embedding=1, safetyCheck=True, GPU=False)

        sparse_tensor = adj.coalesce()
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        for i in range(indices.shape[1]):
            node_from = indices[0, i].item()
            node_to = indices[1, i].item()
            value = values[i].item()
            if not (node_from, node_to) in self.node_weights.keys():
                self.node_weights[(node_from, node_to)] = [value]
            else:
                self.node_weights[(node_from, node_to)].append(value)
            #print(f"Value {value} corresponds to connection from node {node_from} to node {node_to}")

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

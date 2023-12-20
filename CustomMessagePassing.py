import torch
import torch_scatter
from torch import Tensor
from torch_geometric.utils import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from PyIF import te_compute as te
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
#sliding_window_view(np.array([4, 2, 3, 8, -6, 10]), window_shape = 3)
from functools import reduce
from scipy.special import expit

class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix.
        x = self.lin(x)

        # Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # x_i and x_j are features of nodes i and j, where (i, j) is an edge.

        # Here you can apply your custom algorithm to update the weights
        # based on the features x_i and x_j.
        # For example, a simple operation:
        custom_weight = self.custom_weight_update(x_i, x_j)

        # Return the updated features for aggregation.
        #norm = F.normalize((torch.tensor(custom_weight).to(device)).to(torch.float32))
        #tes = torch.tensor(custom_weight).to(device).to(torch.float32)
        return custom_weight * x_j
        #return custom_weight * x_j

    def custom_weight_update(self, x_i, x_j):
        # Implement your custom weight update algorithm here.

        #te1 = te.te_compute(x_i.detach().cpu().numpy().flatten(), x_j.detach().cpu().numpy().flatten(), k=1, embedding=1, safetyCheck=False, GPU=False)
        #te.te_compute(np.array([0,1,0,1,0,0,1,0,1,0,]), np.array([0,1,0,0,0,0,1,0,0,0]), k=1, embedding=1,safetyCheck=False, GPU=True)
        #te1 = te.te_compute(x_i.detach().cpu().numpy().flatten(), x_j.detach().cpu().numpy().flatten(), k=100, embedding=1, safetyCheck=False, GPU=False)
        tes = []
        for i, xi in enumerate(x_i.t().detach().cpu().numpy()):
            teitem = te.te_compute(xi, x_j[:,i].detach().cpu().numpy(), k=1, embedding=1, safetyCheck=False, GPU=False)
            tes.append(teitem)
        #return tes
        return torch.sigmoid(torch.tensor(tes).to(device).to(torch.float32))
        #return expit(torch.tensor(tes).to(device).to(torch.float32) + x_i)
        #return torch.sigmoid(torch.sum(x_i * x_j, dim=-1, keepdim=True))

    def aggregate(self, inputs, index, dim_size=None):
        # The aggregation method. For simplicity, we use summation here.
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        #return torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce='mean', dim_size=dim_size)

    def update(self, inputs: Tensor) -> Tensor:
        return inputs

import torch.nn.functional as F

class CustomGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGCN, self).__init__()
        self.conv1 = CustomConv(in_channels, 16)
        self.conv2 = CustomConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import os.path as osp

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomGCN(dataset.num_node_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)

    #loss = criterion(out[data.train_mask], data.y[data.train_mask])
    #loss.backward()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    optimizer.step()
    return loss.item()

def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

for epoch in range(200):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')


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
from utils import get_baseline_run
import neptune

token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMDVkMzRmYS1hODJlLTQ3OGItYjdiYi0zZTJhMGI1ZGNhOGIifQ=="
pname = "adrian.moldovan/GNN"

run = neptune.init_run(
    project=pname,
    api_token=token,
    source_files=['*.py'],
    #tags=["torch.sigmoid(0.1 * detached * x_j)"] not working!!!!
    #tags = ["sigmoid(0.1 * tes * x_j)"]
    #tags = ["sigmoid(tes * x_j)"]
    tags = ["torch.sigmoid((detached/2.) * x_j)"] #for the 2nd conv layer
    #tags = ["torch.sigmoid(torch.sum(0.1 * detached * x_j, dim=-1, keepdim=True))"] !!!!!!!!! need to test
    #tags=["torch.sigmoid(0.9 * detached * x_j)"] #does not work
    #tags=["torch.sigmoid(detached + torch.sum(x_i * x_j, dim=-1, keepdim=True))"]
    #tags=["torch.sigmoid(detached + torch.sum(x_i * x_j, dim=-1, keepdim=True))"] NOT OK!!!
    # mode='read-only'
)


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
        return custom_weight
        #return custom_weight * x_j

    def custom_weight_update(self, x_i, x_j):
        tes = []
        for i, xi in enumerate(x_i.t().detach().cpu().numpy()):
            teitem = te.te_compute(xi, x_j[:,i].detach().cpu().numpy(), k=1, embedding=1, safetyCheck=False, GPU=False)
            tes.append(teitem)
        #return tes
        detached = torch.tensor(tes).to(device).to(torch.float32)
        #not tried return torch.sigmoid(torch.tensor(tes).to(device).to(torch.float32) + x_j)
        #return torch.sigmoid(detached * x_j)
        #return torch.sigmoid(0.1 * detached * x_j) not ok

        if detached.shape[0] == 7:
            return torch.sigmoid((detached/2.) * x_j)
        return torch.sigmoid(detached * x_j)
        #return torch.sigmoid(torch.sum(0.1 * detached * x_j, dim=-1, keepdim=True))

        #return expit(torch.tensor(tes).to(device).to(torch.float32) + x_i)
        #return torch.sigmoid(torch.sum(x_i * x_j, dim=-1, keepdim=True))
        #return torch.sigmoid(torch.sum(x_i * x_j * detached, dim=-1, keepdim=True))

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

import pandas as pd
dfbaseline = pd.read_csv('./baseline.csv')
dfbaseline.set_index('epoch')

def log(key, val):
    run[key].append(val)

def log_set(dict:dict, baseline=False):
    for k in dict.keys():
        if baseline == True:
            run[k + '_b'].append(dict[k])
        else:
            run[k].append(dict[k])

for epoch in range(200):
    loss = train()
    train_acc, val_acc, test_acc = test()
    log_set({"train_acc" : train_acc, "train_loss": loss, "epoch": epoch, "val_acc" : val_acc, "test_acc" : test_acc})
    log_set(dfbaseline.loc[dfbaseline['epoch'].eq(epoch+1)].to_dict(orient='records')[0], baseline=True)

    # run["train_acc"].append(train_acc)
    # run["epoch"].append(epoch)
    # run["val_acc"].append(val_acc)
    # run["test_acc"].append(test_acc)
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')


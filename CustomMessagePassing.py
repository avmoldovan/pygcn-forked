import torch
import torch_scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

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
        return custom_weight * x_j

    def custom_weight_update(self, x_i, x_j):
        # Implement your custom weight update algorithm here.
        # This is just a placeholder example:
        return torch.sigmoid(torch.sum(x_i * x_j, dim=-1, keepdim=True))

    def aggregate(self, inputs, index, dim_size=None):
        # The aggregation method. For simplicity, we use summation here.
        return torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce='mean', dim_size=dim_size)


import torch.nn.functional as F

class CustomGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGCN, self).__init__()
        self.conv1 = CustomConv(in_channels, 16)
        self.conv2 = CustomConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
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
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
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


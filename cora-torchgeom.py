import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SplineConv
#from torch_geometric.nn import GCNConv
from GCNConv import GCNConv
from torch_geometric.typing import WITH_TORCH_SPLINE_CONV
import matplotlib.pyplot as plt

if not WITH_TORCH_SPLINE_CONV:
    quit("This example requires 'torch-spline-conv'")

dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit(num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid('./data/cora', dataset, transform=transform)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = GCNConv(16, dataset.num_classes, dim=1, kernel_size=2)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

# def find_max_weight_edge(G):
#     max_weight = float('-inf')
#     max_edge = None
#
#     for edge in G.edges(data=True):
#         node1, node2, data = edge
#         weight = data.get('weight', 1)  # Default weight is 1 if not specified
#         if weight > max_weight:
#             max_weight = weight
#             max_edge = (node1, node2)
#
#     return max_edge, max_weight

def train():
    model.train()

    # Find the edge with the highest weight
    #edge_with_highest_weight, weight = find_max_weight_edge(G)

    #print(f"Edge with the highest weight: {edge_with_highest_weight}, Weight: {weight}")

    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

# def evaluate_edge_impact(model, data, edge):
#     # Isolate the edge
#     isolated_edge_index = edge.unsqueeze(1)
#
#     # Perform a forward pass with only this edge
#     output = model()#data.x, isolated_edge_index)
#     return output

def plot_edge_impacts(edge_impact_history, edge_indices):
    """
    Plots the impact of specified edges over epochs.
    :param edge_impact_history: List of edge impacts per epoch.
    :param edge_indices: Indices of edges to plot.
    """
    for edge_idx in edge_indices:
        for epoch_impacts in edge_impact_history[0]:
            impacts = epoch_impacts[edge_idx][1].item()
        #impacts = [epoch_impacts[edge_idx][1].item() for epoch_impacts in edge_impact_history[0]]
        plt.plot(impacts, label=f'Edge {impacts[edge_idx][0]}')

    plt.xlabel('Epoch')
    plt.ylabel('Edge Impact')
    plt.title('Edge Impact Over Epochs')
    plt.legend()
    plt.show()

edge_impact_per_epoch = []
for epoch in range(1, 201):
    train()

    train_acc, test_acc = test()

    epoch_edge_impacts = []

    # if epoch % 25 == 0:
    #     with torch.no_grad():
    #         for i in range(data.edge_index.size(1)):
    #             edge = data.edge_index[:, i]
    #             isolated_edge_index = edge.unsqueeze(1)
    #
    #             # Perform a forward pass with only this edge
    #             impact = model()#data.x, isolated_edge_index)
    #
    #             epoch_edge_impacts.append((edge.detach().clone().cpu(), impact.detach().clone().cpu()))
    #             torch.cuda.empty_cache()  # Free up memory
    #
    #         edge_impact_per_epoch.append(epoch_edge_impacts)
    #         #plot_edge_impacts(edge_impact_per_epoch, edge_indices=[0, 1, 2])  # Plot impacts for the first three edges

    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, n_in_dim=11, n_out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(n_in_dim, 16)
        self.conv2 = GCNConv(16, 16)
        self.output = nn.Linear(16, n_out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # readout layer
        x = global_mean_pool(x, batch=data.batch)
        x = F.relu(x)

        # output layer
        x = self.output(x)
        return x

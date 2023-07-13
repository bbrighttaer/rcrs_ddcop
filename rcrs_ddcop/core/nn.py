import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling


class GCN(torch.nn.Module):
    def __init__(self, n_in_dim=11, n_out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(n_in_dim, 16)
        self.conv2 = GCNConv(16, 16)
        self.pool1 = TopKPooling(16, ratio=0.3)
        self.output = nn.Linear(16, n_out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.pool1(x, edge_index, batch=data.batch)

        # readout layer
        x = global_mean_pool(x, batch=data.batch)
        x = F.relu(x)

        # output layer
        x = self.output(x)
        return x


class NodeGCN(torch.nn.Module):
    def __init__(self, dim=11):
        super().__init__()
        self.conv1 = GCNConv(dim, 16)
        self.conv2 = GCNConv(16, 4)
        self.conv3 = GCNConv(4, 16)
        self.conv4 = GCNConv(16, dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # encoder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # bottleneck
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # decoder
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling
from rcrs_core.log.logger import Logger

from rcrs_ddcop.core.data import process_data
from rcrs_ddcop.core.experience import ExperienceBuffer


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


class train:
    """Model training block"""

    def __init__(self, model: NodeGCN):
        self._model = model

    def __enter__(self):
        self._model.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model.eval()


class ModelTrainer:

    def __init__(self, model: NodeGCN, experience_buffer: ExperienceBuffer, log: Logger, batch_size: int):
        self.model = model
        self.experience_buffer = experience_buffer
        self.batch_size = batch_size
        self.log = log

    def __call__(self, *args, **kwargs):
        """Trains the given model using data from the experience buffer"""

        # ensure there is enough data to sample from
        if len(self.experience_buffer) < self.batch_size * 2:
            return

        self.log.debug('Training initiated...')

        # start training block
        with train(self.model):
            sampled_data = self.experience_buffer.sample(self.batch_size * 2)
            dataset = process_data(sampled_data)
            data_loader = DataLoader(dataset, batch_size=self.batch_size)

            # training loop
            for batch in data_loader:
                ...



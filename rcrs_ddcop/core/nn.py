import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rcrs_core.log.logger import Logger
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling

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

    def __init__(self, label: str, model: NodeGCN, experience_buffer: ExperienceBuffer, log: Logger, batch_size: int,
                 epochs: int = 100, lr=0.01, weight_decay: float = 1e-3, gamma: float = 0, transform=None):
        self.label = label
        self.model = model
        self.experience_buffer = experience_buffer
        self.batch_size = batch_size
        self.log = log
        self.epochs = epochs
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.writer = SummaryWriter()
        self.count = 0
        self.normalizer = transform
        self.is_training = False
        self.scheduler = StepLR(self.optimizer, step_size=epochs, gamma=gamma)

    def __call__(self, *args, **kwargs):
        """Trains the given model using data from the experience buffer"""

        # ensure there is enough data to sample from
        if len(self.experience_buffer) < self.batch_size * 2:
            return

        self.log.debug('Training initiated...')
        self.is_training = True

        # start training block
        with train(self.model):
            sampled_data = self.experience_buffer.sample(self.batch_size * 2)
            dataset = process_data(sampled_data, transform=self.normalizer)
            data_loader = DataLoader(dataset, batch_size=self.batch_size)

            losses = []

            # training loop
            for i in range(self.epochs):
                for batch in data_loader:
                    # reset gradient registries
                    self.optimizer.zero_grad()

                    # normalize data
                    batch.x = torch.tensor(self.normalizer.transform(batch.x), dtype=torch.float)
                    batch.y = torch.tensor(self.normalizer.transform(batch.y), dtype=torch.float)

                    # forward pass
                    output = self.model(batch)

                    # zero-out building code prediction
                    output[:, 3] = 0.
                    batch.y[:, 3] = 0.

                    # loss function
                    loss = self.criterion(output, batch.y)
                    losses.append(loss.item())

                    # backward pass
                    loss.backward()

                    # parameters update
                    self.optimizer.step()

                # self.scheduler.step()

            # metrics
            avg_loss = np.mean(losses)
            # print(avg_loss)
            self.writer.add_scalars('Loss', {self.label: avg_loss}, self.count)
            self.count += 1

        self.is_training = False

        return avg_loss

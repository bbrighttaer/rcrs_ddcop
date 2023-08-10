from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rcrs_core.log.logger import Logger
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback, _Model, EarlyStopping

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


def r2(y_pred: np.ndarray, y_true: xgb.DMatrix) -> Tuple[str, float]:
    """Calculate Correlation of Determination score"""
    y_true = y_true.get_label()
    y_pred = y_pred.ravel()
    score = r2_score(y_true, y_pred)
    return 'r2', score


class XGBTrainer:
    """
    Trains an XGBoost model
    """

    def __init__(self, label: str, model_params: dict, experience_buffer: ExperienceBuffer, log: Logger,
                 transform=None, rounds: int = 100):
        self.label = label
        self.params = model_params
        self.experience_buffer = experience_buffer
        self.log = log
        self.writer = SummaryWriter()
        self.count = 0
        self.normalizer = transform
        self.best_model_config = None
        self.best_score = float('-inf')
        self._model = None
        self._rounds = rounds
        self.is_training = False

    @property
    def model(self):
        return self._model

    def write_to_tf_board(self, name, t, val):
        self.writer.add_scalars(name, {self.label: val}, t)

    def __call__(self, *args, **kwargs):
        """Trains the prediction model"""
        # ensure there is enough data to sample from
        if len(self.experience_buffer) < 10:
            return

        self.log.debug('Training initiated...')
        self.is_training = True

        # start training block
        sampled_data = self.experience_buffer.sample(len(self.experience_buffer))
        dataset = process_data(sampled_data, transform=self.normalizer)
        data = next(iter(DataLoader(dataset, batch_size=len(dataset))))

        # normalize data to zero mean, unit variance
        X = self.normalizer.transform(data.x)
        Y = self.normalizer.transform(data.y)

        # split data
        tr_sz = int(len(X) * 0.8)
        X_train = X[:tr_sz, :]
        Y_train = Y[:tr_sz, :]
        X_val = X[tr_sz:, :]
        Y_val = Y[tr_sz:, :]

        # put data into xgb matrix
        dtrain = xgb.DMatrix(data=X_train, label=Y_train)
        dval = xgb.DMatrix(data=X_val, label=Y_val)

        # train model
        model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self._rounds,
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=[
                ModelCheckpointCallback(self),
                EarlyStopping(
                    rounds=5,
                    metric_name='rmse',
                    data_name='val',
                )
            ],
            custom_metric=r2,
            verbose_eval=False,
            xgb_model=self._model,
        )

        if self.best_model_config:
            model.load_config(self.best_model_config)
            self._model = model

        return model.best_score


class ModelCheckpointCallback(TrainingCallback):

    def __init__(self, trainer: XGBTrainer):
        super().__init__()
        self._trainer = trainer

    def after_iteration(self, model: _Model, epoch: int, evals_log) -> bool:
        r2_val = evals_log['val']['r2'][-1]
        rmse = evals_log['val']['rmse'][-1]
        if r2_val > self._trainer.best_score:
            self._trainer.best_score = r2_val
            self._trainer.best_model_config = model.save_config()

        # report to tensorboard
        self._trainer.writer.add_scalars('r2', {self._trainer.label: r2_val}, self._trainer.count)
        self._trainer.writer.add_scalars('rmse', {self._trainer.label: rmse}, self._trainer.count)
        self._trainer.count += 1

        return False

from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling
import xgboost as xgb
from rcrs_core.log.logger import Logger
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from xgboost.callback import TrainingCallback, _Model, EarlyStopping

from rcrs_ddcop.core.experience import ExperienceBuffer


def save_training_data(label, data, columns, suffix):
    # save data
    df = pd.DataFrame(
        data,
        columns=columns,
    )
    df.to_csv(f'{label}_training_data_{suffix}.csv', index=False)


# class GCN(torch.nn.Module):
#     def __init__(self, n_in_dim=11, n_out_dim=1):
#         super().__init__()
#         self.conv1 = GCNConv(n_in_dim, 16)
#         self.conv2 = GCNConv(16, 16)
#         self.pool1 = TopKPooling(16, ratio=0.3)
#         self.output = nn.Linear(16, n_out_dim)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         # obtain node embeddings
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = self.pool1(x, edge_index, batch=data.batch)
#
#         # readout layer
#         x = global_mean_pool(x, batch=data.batch)
#         x = F.relu(x)
#
#         # output layer
#         x = self.output(x)
#         return x


# class NodeGCN(torch.nn.Module):
#     def __init__(self, dim=11):
#         super().__init__()
#         self.conv1 = GCNConv(dim, 16)
#         self.conv2 = GCNConv(16, 4)
#         self.conv3 = GCNConv(4, 16)
#         self.conv4 = GCNConv(16, dim)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         # encoder
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#
#         # bottleneck
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#
#         # decoder
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv4(x, edge_index)
#
#         return x


class train:
    """Model training block"""

    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model.eval()


class ModelTrainer:

    def __init__(self, label: str, experience_buffer: ExperienceBuffer, log: Logger, batch_size: int,
                 epochs: int = 100, lr=0.01, weight_decay: float = 1e-3, gamma: float = 0, transform=None):
        self.label = label
        self.has_trained = False
        self.model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )
        self.experience_buffer = experience_buffer
        self.sample_size = 500
        self.batch_size = 128
        self.log = log
        self.epochs = epochs
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.writer = SummaryWriter()
        self.count = 0
        self.normalizer = transform
        self.is_training = False
        self.best_score = float('-inf')
        self.scheduler = StepLR(self.optimizer, step_size=epochs, gamma=gamma)
        self.columns = [
            'fieryness_x', 'temperature_x', 'brokenness_x', 'building_code_x', 'fire_index_x',
            'fieryness_y', 'temperature_y', 'brokenness_y', 'building_code_y', 'fire_index_y',
        ]
        # self.load_model()

    def load_model(self):
        model_file_name = f'{self.label}_model.pt'
        scaler_file_name = f'{self.label}_scaler.bin'
        self.model.load_state_dict(torch.load(model_file_name))
        self.model.eval()
        self.has_trained = True
        self.normalizer = joblib.load(scaler_file_name)
        self.log.info('Model loaded successfully')

    def __call__(self, *args, **kwargs):
        """Trains the given model using data from the experience buffer"""

        # ensure there is enough data to sample from
        if len(self.experience_buffer) < 100:
            return

        # self.log.debug('Training initiated...')
        # self.is_training = True
        #
        # # start training block
        # with train(self.model):
        #     dataset = self.experience_buffer.sample(self.sample_size)
        #     dataset = np.array(dataset)
        #     save_training_data(self.label, dataset, self.columns, 'original')
        #     X, Y = dataset[:, :5], dataset[:, 5:]
        #
        #     # normalize data
        #     self.normalizer.fit(np.concatenate([X, Y], axis=0))
        #
        #     # normalize samples
        #     X = self.normalizer.transform(X)
        #     Y = self.normalizer.transform(Y)
        #
        #     # split data
        #     X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)
        #
        #     # convert to tensors
        #     X_train = torch.tensor(X_train).float()
        #     y_train = torch.tensor(y_train).float()
        #     X_val = torch.tensor(X_val).float()
        #     y_val = torch.tensor(y_val).float()
        #
        #     training_losses = []
        #
        #     # train model
        #     for i in range(self.epochs):
        #         # reset gradient registries
        #         self.optimizer.zero_grad()
        #
        #         # forward pass
        #         outputs = self.model(X_train)
        #
        #         # zero-out building code and brokenness prediction
        #         outputs[:, 2:4] = 0.
        #         y_train[:, 2:4] = 0.
        #
        #         # calculate training loss
        #         train_loss = self.criterion(outputs, y_train)
        #         training_losses.append(train_loss.item())
        #
        #         # backward pass
        #         train_loss.backward()
        #
        #         # parameters update
        #         self.optimizer.step()
        #
        #         # self.scheduler.step()
        #
        #     # metrics
        #     avg_loss = np.mean(training_losses)
        #     val_outputs = self.model(X_val).detach()
        #     # zero-out building code and brokenness prediction
        #     val_outputs[:, 2:4] = 0.
        #     y_val[:, 2:4] = 0.
        #     val_loss = self.criterion(val_outputs, y_val)
        #     r2_val = r2_score(val_outputs.numpy(), y_val.numpy())
        #     self.writer.add_scalars('Training loss', {self.label: avg_loss}, self.count)
        #     self.writer.add_scalars('Val loss', {self.label: val_loss.item()}, self.count)
        #     self.writer.add_scalars('r2', {self.label: r2_val}, self.count)
        #
        #     if r2_val > self.best_score:
        #         self.best_score = r2_val
        #         torch.save(self.model.state_dict(), f'{self.label}_model.pt')
        #         joblib.dump(self.normalizer, f'{self.label}_scaler.bin')
        #
        #     self.count += 1
        #
        # self.is_training = False
        #
        # self.has_trained = True

        # return avg_loss

    def write_to_tf_board(self, name, t, val):
        self.writer.add_scalars(name, {self.label: val}, t)


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

    def __init__(self, label: str, experience_buffer: ExperienceBuffer, log: Logger,
                 transform=None, rounds: int = 100):
        self.label = label
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 10,
            'learning_rate': 1e-3,
            'tree_method': 'hist',
            'n_jobs': 16,
            'multi_strategy': 'multi_output_tree'
            # 'reg_lambda': 1e-5,
        }
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
        self.batch_sz = 700

        # self.load_model()

    @property
    def model(self):
        return self._model

    def load_model(self):
        self._model = xgb.Booster()
        model_file_name = f'{self.label}_model_static.json'
        scaler_file_name = f'{self.label}_scaler_static.bin'
        # if 'FireBrigade' in self.label:
        #     model_file_name = 'FireBrigadeAgent_210552869_model.json'
        #     scaler_file_name = 'FireBrigadeAgent_210552869_scaler.bin'
        self._model.load_model(model_file_name)
        self.normalizer = joblib.load(scaler_file_name)
        self.log.info('Model loaded successfully')

    def write_to_tf_board(self, name, t, val):
        self.writer.add_scalars(name, {self.label: val}, t)

    def __enter__(self):
        self.is_training = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_training = False

    def __call__(self, *args, **kwargs):
        """Trains the prediction model"""
        # ensure there is enough data to sample from
        if len(self.experience_buffer) < 10:
            return

        with self:
            self.log.debug('Training initiated...')

            columns = [
                'fieryness_x', 'temperature_x', 'brokenness_x', 'building_code_x', 'fire_index_x',
                'fieryness_y', 'temperature_y', 'brokenness_y', 'building_code_y', 'fire_index_y',
            ]

            # start training block
            dataset = self.experience_buffer.sample(self.batch_sz)
            dataset = np.array(dataset)

            self._save_training_data(dataset, columns, 'original')

            # try:
            #     X, Y = correct_skewed_data(data.x, data.y, columns, 'fieryness_x')
            # except ValueError as e:
            #     self.log.warning(f'Training terminated due to: {str(e)}')
            #     return
            X, Y = dataset[:, :5], dataset[:, 5:]

            # normalize data to zero mean, unit variance
            self.normalizer.fit(np.concatenate([X, Y], axis=0))
            X = self.normalizer.transform(X)
            Y = self.normalizer.transform(Y)
            # wts = np.array([1. if 3 > e > 0 else 0.2 for e in data.y[:, 0]])

            X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)

            # put data into xgb matrix
            dtrain = xgb.DMatrix(data=X_train, label=y_train[:, :-1])
            dval = xgb.DMatrix(data=X_val, label=y_val[:, :-1])

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
                model.save_model(f'{self.label}_model.json')
                joblib.dump(self.normalizer, f'{self.label}_scaler.bin')

        return self.best_score


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

import math
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from xgboost.callback import EarlyStopping

from rcrs_ddcop.core.data import trajectories_to_supervised
from rcrs_ddcop.core.nn import r2, ModelCheckpointCallback


def sample_z(mu, log_sigma):
    eps = torch.randn_like(log_sigma)
    return mu + eps * torch.exp(0.5 * log_sigma)


def kl_divergence_diagonal_gaussians_stable(posterior_mu, posterior_logvar, prior_mu, prior_logvar):
    # Compute the terms for the KL divergence formula
    var_posterior = torch.exp(posterior_logvar)
    var_prior = torch.exp(prior_logvar)
    logvar_diff = prior_logvar - posterior_logvar
    mean_diff = posterior_mu - prior_mu

    # KL divergence calculation
    kl_divergence = 0.5 * torch.sum(var_posterior / var_prior + logvar_diff - 1 + (mean_diff.pow(2) / var_prior))

    return kl_divergence


class RNNEncoder(nn.Module):
    """
    Encodes observations of a trajectory to incorporate historical information.
    """

    def __init__(self, traj_len, input_dim, rnn_hidden_dim, output_dim):
        super(RNNEncoder, self).__init__()
        self.traj_len = traj_len
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, output_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state)
        q = F.relu(self.fc2(h))
        return h, q


class VariationalEncoder(nn.Module):

    def __init__(self, latent_dim, input_dim, h_dim):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, h_dim)
        self.linear_mu = nn.Linear(h_dim, latent_dim)
        self.linear_sigma = nn.Linear(h_dim, latent_dim)

        self.N = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear_mu(x)
        log_sigma = self.linear_sigma(x)
        z = sample_z(mu, log_sigma)
        return z, mu, log_sigma


class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim, h_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, h_dim)
        self.linear2 = nn.Linear(h_dim, output_dim)

    def forward(self, z):
        x = F.leaky_relu(self.linear1(z))
        x = F.relu(self.linear2(x))
        return x


class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dim, input_dim, h_dim, output_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dim, input_dim, h_dim)
        self.decoder = Decoder(latent_dim, output_dim, h_dim)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x = self.decoder(z)
        return x, z, mu, log_sigma

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x


class ImaginationModel:

    def __init__(self, label, traj_len, input_dim, latent_dim, obs_encoding_dim, tb_writer):
        super().__init__()
        self.label = label
        self.traj_len = traj_len
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.rnn_encoder = RNNEncoder(
            traj_len=traj_len,
            input_dim=3,
            rnn_hidden_dim=obs_encoding_dim,
            output_dim=obs_encoding_dim,
        )
        self.posterior_vae = VariationalAutoencoder(
            input_dim=obs_encoding_dim,
            latent_dim=latent_dim,
            h_dim=latent_dim * 2,
            output_dim=input_dim,
        )
        self.prior_vae = VariationalAutoencoder(
            input_dim=latent_dim,
            latent_dim=latent_dim,
            h_dim=latent_dim * 2,
            output_dim=latent_dim
        )
        self.fcn = nn.Sequential(
            nn.Linear(obs_encoding_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        self.normalizer = None
        self.tb_writer = tb_writer
        self.hidden_state = None
        self.best_score = float('-inf')
        self.count = 0
        self.best_model_config = None
        self.xgb_model = None

    @property
    def writer(self):
        return self.tb_writer

    def init_hidden(self, batch_size):
        h0 = self.rnn_encoder.init_hidden()
        h0 = h0.squeeze().expand(batch_size, -1)
        return h0

    def train_xgboost(self, dataset, lr, num_epochs, gamma, batch_size, weight_decay):
        score = 0.
        dataset = dataset.reshape(-1, 5)[:, :self.input_dim]
        normalizer = StandardScaler()
        dataset = normalizer.fit_transform(dataset)
        dataset = dataset.reshape(-1, self.input_dim * self.traj_len)
        self.normalizer = normalizer

        X, y = trajectories_to_supervised(
            dataset=dataset,
            in_dim=self.input_dim,
            past_window_size=3,
            future_window_size=1,
        )


        params = {
            'objective': 'reg:squarederror',
            'max_depth': 10,
            'learning_rate': 1e-3,
            'tree_method': 'hist',
            'n_jobs': 16,
            'multi_strategy': 'multi_output_tree'
            # 'reg_lambda': 1e-5,
        }

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

        # put train_data into xgb matrix
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=4000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=[
                ModelCheckpointCallback(self),
                EarlyStopping(
                    rounds=100,
                    metric_name='rmse',
                    data_name='val',
                )
            ],
            custom_metric=r2,
            verbose_eval=True,
            # xgb_model=self._model,
        )

        if self.best_model_config:
            model.load_config(self.best_model_config)
            self.xgb_model = model
            model.save_model(f'{self.label}_model.json')
            joblib.dump(self.normalizer, f'{self.label}_scaler.bin')

        return score

    def train_fcn(self, dataset, lr, num_epochs, gamma, batch_size, weight_decay):
        params = list(self.rnn_encoder.parameters()) + list(self.fcn.parameters())
        optimizer = Adam(params, lr=lr, weight_decay=0)
        criterion = MSELoss()
        normalizer = StandardScaler()
        # scheduler = StepLR(optimizer, step_size=10, gamma=gamma)

        # preprocess data
        dataset = dataset.reshape(-1, self.input_dim)
        dataset = normalizer.fit_transform(dataset)
        self.normalizer = normalizer
        dataset = dataset.reshape(-1, self.traj_len * self.input_dim)

        # split dataset
        traj_data_train, traj_data_val = train_test_split(dataset, test_size=0.2, shuffle=True)
        c_data = np.concatenate([traj_data_train[:1, :], traj_data_val], axis=0)
        train_data = torch.from_numpy(traj_data_train.reshape(self.traj_len, -1, self.input_dim)).float()
        val_data = torch.from_numpy(traj_data_val.reshape(self.traj_len, -1, self.input_dim)).float()
        # np.savetxt('val_data.csv', normalizer.inverse_transform(val_data.reshape(-1, self.input_dim)), delimiter=',')
        # np.savetxt('train_data.csv', normalizer.inverse_transform(traj_data_train[:2, :].reshape(-1, self.input_dim)), delimiter=',')

        # training loop
        max_score = float('-inf')
        num_batches = math.ceil(train_data.shape[1] / batch_size)
        for e in range(num_epochs):
            losses = []
            for i in range(num_batches):
                offset = i * batch_size
                batch = train_data[:, offset: offset + batch_size, :]
                loss = self._batch_train_fcn(batch, optimizer, criterion, e)
                losses.append(loss)

            # loss computations
            e_loss = np.mean(losses)

            # logging
            print(f'Epoch {e + 1}, Loss = {e_loss}')
            self.tb_writer.add_scalar('fcn training loss', e_loss, e)

            # metrics
            # train_loss, train_r2 = self.validate_fcn(train_data)
            # self.tb_writer.add_scalar('fcn t loss', train_loss, e)
            # self.tb_writer.add_scalar('fcn training r2', train_r2, e)
            val_loss, val_r2 = self.validate_fcn(val_data)
            self.tb_writer.add_scalar('fcn val loss', val_loss, e)
            self.tb_writer.add_scalar('fcn val r2', val_r2, e)

            # save best model
            if val_r2 > max_score:
                # self.save_fcn_models()
                max_score = val_r2

            # scheduler.step()
        return max_score

    def train_model(self, train_data, lr, num_epochs, batch_size, gamma):
        params = list(self.rnn_encoder.parameters())
        params += list(self.posterior_vae.parameters()) + list(self.prior_vae.parameters())
        optimizer = Adam(params, lr=lr, weight_decay=1e-4)
        criterion = MSELoss()
        normalizer = StandardScaler()
        # scheduler = StepLR(optimizer, step_size=10, gamma=gamma)

        # preprocess data
        train_data = train_data.reshape(-1, self.input_dim)
        train_data = normalizer.fit_transform(train_data)
        train_data = train_data.reshape(self.traj_len, -1, self.input_dim)
        train_data = torch.from_numpy(train_data).float()
        self.normalizer = normalizer

        # training loop
        min_loss = float('inf')
        num_batches = math.ceil(train_data.shape[1] / batch_size)
        for e in range(num_epochs):
            losses = []
            losses_q_mse = []
            losses_p_mse = []
            losses_pkl = []
            losses_kl_pq = []
            for i in range(num_batches):
                offset = i * batch_size
                batch = train_data[:, offset: offset + batch_size, :]
                loss, q_mse, p_mse, prior_kl, kl_pq = self._batch_train(batch, optimizer, criterion)
                losses.append(loss)
                losses_q_mse.append(q_mse)
                losses_p_mse.append(p_mse)
                losses_pkl.append(prior_kl)
                losses_kl_pq.append(kl_pq)

            # loss computations
            e_loss = np.mean(losses)
            q_mse = np.mean(losses_q_mse)
            p_mse = np.mean(losses_p_mse)
            prior_kl = np.mean(losses_pkl)
            kl_pq = np.mean(losses_kl_pq)

            # logging
            print(f'Epoch {e + 1}, Loss = {e_loss}')
            self.tb_writer.add_scalar('imagination loss', e_loss, e)
            self.tb_writer.add_scalar('posterior mse', q_mse, e)
            self.tb_writer.add_scalar('prior mse', p_mse, e)
            self.tb_writer.add_scalar('prior kl', prior_kl, e)
            self.tb_writer.add_scalar('posterior-prior kl', kl_pq, e)

            # save best model
            if e_loss < min_loss:
                self.save_models()
                min_loss = e_loss

            # scheduler.step()

    def _batch_train(self, batch, optimizer, criterion):
        """
        Performs optimization over a batch.
        Dimension of batch is L, N, H_out
        :param batch:
        :return: the loss
        """
        bz = batch.shape[1]
        optimizer.zero_grad()

        # encode trajectories
        encoded_obs = []
        hs = self.init_hidden(bz)
        for t in range(self.traj_len):
            obs = batch[t, :, :]
            hs, q = self.rnn_encoder(obs, hs)
            encoded_obs.append(q)
        encoded_obs = torch.concat(encoded_obs, dim=0).float()  # dimension = (L*N, H_out)

        # posterior forward pass
        q_x, q_z, q_mu, q_log_sigma = self.posterior_vae(encoded_obs)

        # prior forward pass
        p_x, p_z, p_mu, p_log_sigma = self.prior_vae(q_z)

        # loss computation
        q_y = batch.reshape(-1, batch.shape[-1])
        posterior_recon = criterion(q_x, q_y)
        prior_recon = criterion(q_z, p_z)
        # prior_kl = 0.5 * torch.sum(torch.exp(p_log_sigma) + torch.square(p_mu) - 1. - p_log_sigma)
        prior_kl = kl_divergence_diagonal_gaussians_stable(
            posterior_mu=p_mu,
            posterior_logvar=p_log_sigma,
            prior_mu=torch.zeros_like(p_mu),  # Mean of the normal distribution
            prior_logvar=torch.zeros_like(p_log_sigma),  # Log-variance (or variance) of the normal distribution
        )
        kl_pq_q_detached = kl_divergence_diagonal_gaussians_stable(
            posterior_mu=q_mu.detach(),
            posterior_logvar=q_log_sigma.detach(),
            prior_mu=p_mu,
            prior_logvar=p_log_sigma,
        )
        kl_pq_p_detached = kl_divergence_diagonal_gaussians_stable(
            posterior_mu=q_mu,
            posterior_logvar=q_log_sigma,
            prior_mu=p_mu.detach(),
            prior_logvar=p_log_sigma.detach(),
        )
        alpha = 0.9
        kl_balanced = alpha * kl_pq_p_detached + (1 - alpha) * kl_pq_q_detached
        loss = alpha * posterior_recon + (1 - alpha) * prior_recon  # + prior_kl # + kl_balanced)

        # optimization
        loss.backward()
        optimizer.step()

        return loss.item(), posterior_recon.item(), prior_recon.item(), prior_kl.item(), kl_balanced.item()

    def _batch_train_fcn(self, batch, optimizer, criterion, epoch):
        """
        Performs optimization over a batch.
        Dimension of batch is L, N, H_out
        :param batch:
        :return: the loss
        """
        bz = batch.shape[1]
        optimizer.zero_grad()

        # encode trajectories
        encoded_obs = []
        expected = []
        hs = self.init_hidden(bz)
        for t in range(self.traj_len - 1):
            obs = batch[t, :, :]
            expected.append(batch[t + 1, :, 0])
            x = self.get_fcn_x(obs)
            hs, q = self.rnn_encoder(x, hs)
            encoded_obs.append(q)
        encoded_obs = torch.concat(encoded_obs, dim=0).float()  # dimension = (L*N, H_out)
        expected = torch.concat(expected, dim=0).view(-1, 1)

        # fcn forward pass
        outputs = self.fcn(encoded_obs)

        # loss computation
        loss = criterion(outputs, expected)

        # optimization
        loss.backward()
        optimizer.step()

        return loss.item()

    def save_models(self):
        torch.save(self.rnn_encoder.state_dict(), f'{self.label}_rnn_encoder.pt')
        torch.save(self.posterior_vae.state_dict(), f'{self.label}_posterior_vae.pt')
        torch.save(self.prior_vae.state_dict(), f'{self.label}_prior_vae.pt')
        joblib.dump(self.normalizer, f'{self.label}_scaler.bin')

    def save_fcn_models(self):
        torch.save(self.rnn_encoder.state_dict(), f'{self.label}_rnn_encoder.pt')
        torch.save(self.fcn.state_dict(), f'{self.label}_fcn.pt')
        joblib.dump(self.normalizer, f'{self.label}_scaler.bin')

    def load_models(self):
        self.rnn_encoder.load_state_dict(torch.load(f'{self.label}_rnn_encoder.pt'))
        self.posterior_vae.load_state_dict(torch.load(f'{self.label}_posterior_vae.pt'))
        self.prior_vae.load_state_dict(torch.load(f'{self.label}_prior_vae.pt'))
        self.normalizer = joblib.load(f'{self.label}_scaler.bin')

    def load_fcn_models(self):
        self.rnn_encoder.load_state_dict(torch.load(f'{self.label}_rnn_encoder.pt'))
        self.fcn.load_state_dict(torch.load(f'{self.label}_fcn.pt'))
        self.normalizer = joblib.load(f'{self.label}_scaler.bin')

    def load_xgb_model(self):
        self.xgb_model = xgb.Booster()
        model_file_name = f'{self.label}_model.json'
        scaler_file_name = f'{self.label}_scaler.bin'
        self.xgb_model.load_model(model_file_name)
        self.normalizer = joblib.load(scaler_file_name)
        print('xgboost models load')

    def predict_xg_boost(self, x):
        self.load_xgb_model()
        dim = len(x)
        x = self.normalizer.transform(x.reshape(-1, self.input_dim))
        x = x.reshape(-1, dim)
        output = self.xgb_model.predict(xgb.DMatrix(x))
        # output = output.repeat(3)
        # output = self.normalizer.inverse_transform(output.reshape(-1, self.input_dim))
        output = self.normalizer.inverse_transform(output)
        # output = output.reshape(-1, 3)[:, 0]
        # output = output.reshape(1, -1)
        return output

    @torch.no_grad()
    def predict(self, x, steps) -> np.array:
        # normalize data
        normalized_x = self.normalizer.transform(x).reshape(1, -1, x.shape[-1])
        normalized_x = torch.from_numpy(normalized_x).float()
        print(f'normalized x = {normalized_x}')

        # encode observation
        encoded_obs = []
        if self.hidden_state is None:
            self.hidden_state = self.init_hidden(normalized_x.shape[1])
        seq_len = normalized_x.shape[0]
        for t in range(seq_len):
            obs = normalized_x[t, :, :]
            self.hidden_state, q = self.rnn_encoder(obs, self.hidden_state)
            encoded_obs.append(q)
        encoded_obs = torch.concat(encoded_obs, dim=0).float()  # dimension = (L*N, H_out)
        print(f'encoded obs = {encoded_obs}')

        # encode into latent space
        x, z, _, _ = self.posterior_vae(encoded_obs)

        # imagined steps
        # for _ in range(steps):
        #     z = self.prior_vae.encode(z)

        print(f'predicted z = {z}')

        # decode from latent space
        x_tensor = self.posterior_vae.decode(z)
        print(f'posterior decode = {x_tensor}')
        output = x_tensor.detach().numpy()

        # revert normalization
        output = self.normalizer.inverse_transform(output)
        output = np.clip(output, a_min=0., a_max=None)
        return output

    @torch.no_grad()
    def validate_fcn(self, val_data):
        # encode observation
        encoded_obs = []
        expected = []
        hidden_state = self.init_hidden(val_data.shape[1])
        seq_len = val_data.shape[0]
        for t in range(seq_len - 1):
            obs = val_data[t, :, :]
            expected.append(val_data[t + 1, :, 0])
            x = self.get_fcn_x(obs)
            hidden_state, q = self.rnn_encoder(x, hidden_state)
            encoded_obs.append(q)

        encoded_obs = torch.concat(encoded_obs, dim=0).float()  # dimension = (L*N, H_out)
        expected = torch.concat(expected, dim=0).float().view(-1, 1)

        # fcn forward pass
        outputs = self.fcn(encoded_obs)

        # loss computation
        loss = F.mse_loss(outputs, expected)
        r2_val = r2_score(
            expected.numpy(),
            outputs.numpy(),
        )
        return loss, r2_val

    def get_fcn_x(self, obs):
        p = lambda x: np.reshape(x, (-1, 1))
        return torch.concatenate([p(obs[:, 0]), p(obs[:, 1]), p(obs[:, 4])], dim=1)

    @torch.no_grad()
    def predict_fcn(self, x, steps) -> np.array:
        # normalize data
        normalized_x = self.normalizer.transform(x).reshape(1, -1, x.shape[-1])
        normalized_x = torch.from_numpy(normalized_x).float()
        print(f'normalized x = {normalized_x}')

        # encode observation
        encoded_obs = []
        hidden_state = self.init_hidden(normalized_x.shape[1])
        obs = normalized_x[0, :, :]
        for t in range(steps):
            hidden_state, obs = self.rnn_encoder(obs, hidden_state)
            obs = self.fcn(obs)

        # revert normalization
        output = obs.detach().numpy()
        output = self.normalizer.inverse_transform(output)
        output = np.clip(output, a_min=0., a_max=None)
        return output


def objective_function(args):
    lr, bz, weight_decay = float(args[0]), int(args[1]), float(args[2])
    print(f'lr={lr}, bz={bz}, weight decay={weight_decay}')
    score = train(lr, weight_decay)
    return score


def transform_trajectory_data(traj, feat_dim):
    X = []
    y = []
    for row in traj:
        row = row.reshape(-1, feat_dim)
        X.append(row[0:-1, :])
        y.append(row[1:, 0].reshape(-1, 1))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def train(lr=0.001, weight_decay=0.):
    # global traj_len, batch_size, input_dim, x
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    traj_len = 100
    batch_size = 128
    input_dim = 5
    latent_dim = 256
    obs_encoding_dim = 256
    num_epochs = 300
    gamma = 0.1
    mode = 'eval'

    # load data
    traj_data = np.loadtxt('../../FireBrigadeAgent_210552869.csv', delimiter=',', dtype=float)
    for _ in range(10):
        np.random.shuffle(traj_data)

    writer = SummaryWriter()

    # create model
    model = ImaginationModel(
        label='imagination_poc',
        traj_len=traj_len,
        input_dim=input_dim,
        latent_dim=latent_dim,
        obs_encoding_dim=obs_encoding_dim,
        tb_writer=writer,
    )
    if mode == 'train':
        # train model
        score = model.train_xgboost(
            traj_data,
            lr=lr,
            num_epochs=num_epochs,
            gamma=gamma,
            batch_size=batch_size,
            weight_decay=weight_decay
        )
        return score
    else:
        # model.load_fcn_models()
        # x = traj_data[0, :5].reshape(1, -1)
        x_obs = np.array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   2.,
         0., 328.,   0.,   0.,   4.,   0., 399.,   0.,   0.])
        print(f'x_obs={x_obs}')
        x = x_obs[:15]
        for i in range(10):
            x_new = model.predict_xg_boost(x)
            x = np.concatenate([x[5:], x_new.ravel()])
            print(f'new x: {x}')
        # x = x.reshape(-1, 3)[:, 0].reshape(1, -1)
        print(f'predicted = {x_new.squeeze().tolist()}')


if __name__ == '__main__':
    mode = 'normal'
    if mode == 'hopt':
        hparam_search = {
            'lr': Real(1e-6, 1e-1, prior='log-uniform'),
            'batch_size': Categorical([4, 16, 32, 64, 128, 256]),
            'weight_decay': Real(1e-6, 1e-1),
        }

        res = gp_minimize(objective_function, hparam_search.values(), n_calls=100)
        print(f'Best score={res.fun:.4f}, lr={res.x[0]}, batch size={res.x[1]}, weigh_decay={res.x[2]}')
    else:
        train()

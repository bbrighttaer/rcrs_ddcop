import math
import random

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import no_grad
from torch.utils.tensorboard import SummaryWriter

seed = 0
random.seed(seed)
torch.manual_seed(seed)


class VariationalEncoder(nn.Module):

    def __init__(self, latent_dim: int, input_dim: int):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.bn = nn.BatchNorm1d(num_features=64)
        self.linear_mu = nn.Linear(64, latent_dim)
        self.linear_sigma = nn.Linear(64, latent_dim)

        self.N = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear_mu(x)
        log_sigma = self.linear_sigma(x)
        eps = self.N.sample(mu.shape).squeeze()
        z = mu + torch.exp(log_sigma / 2) * eps
        v = torch.exp(log_sigma) + torch.square(mu) - 1. - log_sigma
        self.kl = 0.5 * torch.sum(v)
        return z


class Decoder(nn.Module):

    def __init__(self, latent_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 64)
        self.linear2 = nn.Linear(64, output_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dim: int, input_dim: int, output_dim: int):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dim, input_dim)
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z


def train(autoencoder: VariationalAutoencoder, train_data: np.array, output_dim, tb_writer, epochs: int = 2000):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=0.1)
    batch_size = 256
    num_batches = math.ceil(train_data.shape[0] / batch_size)
    count = 0
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        for i in range(num_batches):
            offset = i * batch_size
            x = train_data[offset: offset + batch_size, :]
            x = torch.tensor(x).float()
            opt.zero_grad()
            x_hat = autoencoder(x)
            x_true = x[:, : output_dim]
            val_loss = criterion(x_hat, x_true)
            loss = val_loss + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            print(loss.item())
            tb_writer.add_scalar('vae loss', loss.item(), count)
            count += 1

    return autoencoder


if __name__ == '__main__':
    samples = pd.read_csv('FireBrigadeAgent_210552869_training_data_original.csv').to_numpy()
    writer = SummaryWriter()
    data = samples[:, :5]
    latent_dim = 4
    data_dim = data.shape[-1]
    output_dim = 5

    # normalization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    num_features = 5

    is_train = False
    vae = VariationalAutoencoder(
        latent_dim=latent_dim,
        input_dim=data_dim,
        output_dim=output_dim,
    )
    # vae = nn.Sequential(
    #     nn.Linear(num_features, 10),
    #     nn.ReLU(),
    #     nn.Linear(10, num_features),
    # )
    if is_train:
        vae = train(autoencoder=vae, train_data=data, output_dim=output_dim, tb_writer=writer)
        print('done!')

        torch.save(vae.state_dict(), f'vae_poc_model.pt')
        joblib.dump(scaler, f'vae_poc_scaler.bin')
    else:
        with torch.no_grad():
            vae.load_state_dict(torch.load('vae_poc_model.pt'))
            scaler = joblib.load('vae_poc_scaler.bin')

            # predict
            st = samples[0].reshape(1, -1)
            s = st[:, :5]
            s = scaler.transform(s)
            state = st[:, :5]
            n_state = st[:, 5:]
            print(f'{state}, {n_state}')
            x = s
            for i in range(20):
                x = torch.from_numpy(s).float()
                x = vae(x)
                # x = torch.concat([x, torch.zeros((1, 37))], dim=1)
                x = scaler.inverse_transform(x.numpy())
                print(x[:, :5])





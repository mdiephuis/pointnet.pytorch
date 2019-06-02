from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Conv1dBR(nn.Module):
    def __init__(self, input_shape, output_shape, k_size=1):
        super(Conv1dBR, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.encoder = nn.ModuleList([
            nn.Conv1d(input_shape, output_shape, k_size),
            nn.BatchNorm1d(output_shape),
            nn.ReLU()
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x


class Conv1TransBR(nn.Module):
    def __init__(self, input_shape, output_shape, k_size=1):
        super(Conv1TransBR, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.encoder = nn.ModuleList([
            nn.ConvTranspose1d(input_shape, output_shape, k_size),
            nn.BatchNorm1d(output_shape),
            nn.ReLU()
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x


class LinearBR(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LinearBR, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.encoder = nn.ModuleList([
            nn.Linear(input_shape, output_shape),
            nn.BatchNorm1d(output_shape),
            nn.ReLU()
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x


def stn_encoder_decoder(input_shape, encoder_size, decoder_size, latent_size):
    encoder = nn.ModuleList([
        Conv1dBR(input_shape, encoder_size // 64),
        Conv1dBR(encoder_size // 64, encoder_size // 32),
        Conv1dBR(encoder_size // 32, encoder_size),
        LinearBR(encoder_size, encoder_size // 2),
        LinearBR(encoder_size // 2, encoder_size // 4),
    ])

    decoder = nn.ModuleList([
        LinearBR(latent_size, decoder_size // 4),
        LinearBR(decoder_size // 4, decoder_size // 2),
        LinearBR(decoder_size // 2, decoder_size),
        Conv1TransBR(decoder_size, decoder_size // 32),
        Conv1TransBR(decoder_size // 32, decoder_size // 64),
        Conv1TransBR(decoder_size // 64, input_shape),
    ])

    return encoder, decoder


class STNVAE(nn.Module):
    def __init__(self, input_shape, encoder_size, decoder_size, latent_size):
        super(STNVAE, self).__init__()
        self.input_shape = np.prod(list(input_shape))
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.latent_size = latent_size

        self.encoder, self.decoder = stn_encoder_decoder(self.input_shape, self.encoder_size,
                                                         self.decoder_size, self.latent_size)

        self.encoder_mu = LinearBR(self.encoder_size // 4, self.latent_size)

        self.encoder_std = nn.ModuleList([
            LinearBR(encoder_size // 4, latent_size),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.)
        ])

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)

        mu = self.encoder_mu(x)

        std = x
        for layer in self.encoder_std:
            std = layer(std)

        return mu, std

    def decode(self, z):
        x_hat = z
        for layer in self.decoder:
            x_hat = layer(x_hat)

        return x_hat

    def reparameterize(self, mu, std):
        draw = torch.randn_like(std)
        z = draw.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu, std = self.encode(x.view(-1, self.input_shape))
        z = self.reparameterize(mu, std)
        x_hat = self.decode(z)
        return x_hat, mu, std

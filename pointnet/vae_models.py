from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
from torch.nn import init


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)


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
    conv_encoder = nn.ModuleList([
        Conv1dBR(input_shape, encoder_size // 16),
        Conv1dBR(encoder_size // 16, encoder_size // 8),
        Conv1dBR(encoder_size // 8, encoder_size),
    ])

    lin_encoder = nn.ModuleList([
        LinearBR(encoder_size, encoder_size // 2),
        LinearBR(encoder_size // 2, encoder_size // 4),
    ])

    decoder = nn.ModuleList([
        LinearBR(latent_size, decoder_size // 4),
        LinearBR(decoder_size // 4, decoder_size // 2),
        LinearBR(decoder_size // 2, decoder_size),
        Conv1TransBR(decoder_size, decoder_size // 8),
        Conv1TransBR(decoder_size // 8, decoder_size // 16),
        Conv1TransBR(decoder_size // 16, input_shape),
    ])

    return conv_encoder, lin_encoder, decoder


class STNVAE(nn.Module):
    def __init__(self, input_shape, encoder_size, decoder_size, latent_size):
        super(STNVAE, self).__init__()
        if isinstance(input_shape, list) or isinstance(input_shape, tuple):
            self.input_shape = np.prod(list(input_shape))
        else:
            self.input_shape = input_shape
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.latent_size = latent_size

        self.conv_encoder, self.lin_encoder, self.decoder = stn_encoder_decoder(
            self.input_shape, self.encoder_size, self.decoder_size, self.latent_size)

        self.encoder_mu = nn.ModuleList([
            LinearBR(self.encoder_size // 4, self.latent_size),
            nn.ReLU()
        ])

        self.encoder_std = nn.ModuleList([
            LinearBR(encoder_size // 4, latent_size),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.)
        ])

    def encode(self, x):

        for layer in self.conv_encoder:
            x = layer(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, encoder_size)

        for layer in self.lin_encoder:
            x = layer(x)

        mu = x
        for layer in self.encoder_mu:
            mu = layer(mu)

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


if __name__ == '__main__':
    x = torch.randn(32, 3, 2500)
    input_shape = [3]
    encoder_size = 1024
    decoder_size = 1024
    latent_size = 9

    model = STNVAE(input_shape, encoder_size, decoder_size, latent_size)
    model.apply(init_weights)

    x_hat, z_mu, z_std = model(x)

    print('Output x_hat size: {}'.format(x_hat.size()))

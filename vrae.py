import torch

import torch.nn as nn
from torch.nn.init import xavier_uniform as xavier_initializer
from torch.nn import Linear
import torch.nn.functional as F
import torch.optim as optim


def flatten(input):
    return input.view(input.size(0), -1)

class ConvEncoder(nn.Module):

    def __init__(self, in_shape=None, out_dim=20, activation_function=F.relu):
        # assert input_shape is not None
        super(ConvEncoder, self).__init__()
        self.activation_function = activation_function
        self.in_shape = in_shape
        self.latent_dim = out_dim
        self.conv1 = nn.Conv2d(in_channels=in_shape, out_channels=20, kernel_size=3)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 20, 3)
        self.fc1 = nn.Linear(1, self.latent_dim)  # TODO

    def forward(self, x):
        c1 = self.activation_function(self.conv1(x))
        c2 = self.activation_function(self.conv2(c1))
        f = flatten(c2)
        out = self.activation_function(self.fc1(f))
        return out


class ConvDecoder(nn.Module):

    def __init__(self, in_dim, out_shape, activation_functino=F.relu, output_activation=F.sigmoid):
        super(ConvDecoder, self).__init__()
        self.activation_function = activation_functino
        self.output_activation = output_activation
        self.out_shape = out_shape
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 784)  # TODO deconvolution convtranspose2d

    def forward(self, z):
        f1 = self.activation_function(self.fc1(z))
        f2 = self.activation_function(self.fc2(f1))
        x_hat = self.output_activation(self.fc3(f2))
        return x_hat.view(self.out_shape)


class bVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, beta=1., activation_function=F.relu, optimizer=optim.Adam):
        super(bVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta = beta
        self.mu = nn.Linear(encoder.latent_dim, self.latent_dim)
        self.log_var = nn.Linear(encoder.latent_dim, self.latent_dim)
        self.activation_function = activation_function
        self.optimizer = optimizer(params=self.parameters())

    def sample(self, mu, log_var):
        dim = self.latent_dim
        eps = torch.rand(dim)
        return mu + torch.mul(eps, torch.exp(log_var / 2))

    def forward(self, x):
        mu_, log_var_ = self.encode(x)
        z = self.sample(mu_, log_var_)
        out = self.decode(z)
        return out, (mu_, log_var_)

    def encode(self, x):
        # convert x to flat vector
        x_ = self.encoder.forward(x)
        # calc mus and log_var from input
        mu_ = self.activation_function(self.mu(x_))
        log_var_ = self.activation_function(self.log_var(x_))
        return mu_, log_var_

    def decode(self, z):
        return self.decoder.forward(z)

    def bvae_loss(self, y_pred, y, z_mu, z_var):
        recon_loss = F.binary_cross_entropy(y_pred, y, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)
        loss = recon_loss + self.beta*kl_loss
        return loss

    def fit(self, data, labels, n_epochs=100):
        history = []
        if type(data) is not torch.Tensor:
            data = torch.tensor(data)
            labels = torch.tensor(labels)
        for i in range(n_epochs):
            x, y = data[i], labels[i]

            y_pred, (z_mu, z_var) = self.forward(x)

            loss = self.bvae_loss(y_pred, y, z_mu, z_var)

            loss.backward()

            self.optimizer.step()
        return history

    def predict(self, x):
        # run each point in x through forward step but only keep out and throw away (mu, var)
        return torch.tensor([(self.forward(sample))[0] for sample in x])

if __name__ == '__main__':
    print("Testing bVAE")

    print("import")
    from tensorflow.keras.datasets import mnist
    import numpy as np
    print("load data")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, 1) / 255
    x_test = np.expand_dims(x_test, 1) / 255
    data_shape = tuple(x_train.shape[1:])

    print("build encoder/ decoder")
    latent_dim = 32
    encoder = ConvEncoder(in_shape=data_shape, out_dim=64)  # out_dim == dim of mu and dim of log_var
    decoder = ConvDecoder(in_dim=latent_dim, out_shape=data_shape)
    print(encoder.extra_repr())
    print(decoder.extra_repr())

    print("build beta vae")
    bvae = bVAE(encoder, decoder, latent_dim=latent_dim)
    print(bvae.extra_repr())
    print("fit bvae")
    history = bvae.fit(x_train, x_train)
    print("test bvae")
    test_out = bvae.predict(x_test[:25])

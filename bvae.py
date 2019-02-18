import os
import sys
import operator
import functools



import torch

import torch.nn as nn
from torch.nn.init import xavier_uniform as xavier_initializer
from torch.nn import Linear
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.utils import _pair


def calc_conv_out_dims(h_in, w_in, kernel, c_out=None, stride=1, padding=0, dilation=1):
    '''
        stride, padding, dilation set to standard values for torch.nn.Conv2d
        https://pytorch.org/docs/stable/nn.html#conv2d
    '''
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel = _pair(kernel)
    stride = _pair(stride)

    h_out = ((h_in + 2*padding[0] - dilation[0]*(kernel[0] - 1) -1)*(1/stride[0])) + 1
    w_out = ((w_in + 2*padding[1] - dilation[1]*(kernel[1] - 1) -1)*(1/stride[1])) + 1
    assert h_out % 1 == 0
    assert w_out % 1 == 0
    h_out = int(h_out)
    w_out = int(w_out)
    if c_out is None:  # this looks simpler than a one liner
        return h_out, w_out
    else:
        return c_out, h_out, w_out


def flatten(inp):
    return inp.view(inp.size(0), -1)


class ConvEncoder(nn.Module):

    def __init__(self, in_shape=(1, 1, 1), out_dim=20, activation_function=F.relu):
        # assert input_shape is not None
        super(ConvEncoder, self).__init__()
        self.activation_function = activation_function
        self.in_shape = in_shape
        self.latent_dim = out_dim
        self.conv1 = nn.Conv2d(in_channels=in_shape[0], out_channels=40, kernel_size=3)#.to(torch.double)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=10, kernel_size=3)
        c1_out = calc_conv_out_dims(h_in=in_shape[1], w_in=in_shape[2], kernel=3, c_out=40)
        c2_out = calc_conv_out_dims(h_in=c1_out[1], w_in=c1_out[2], c_out=10, kernel=3)
        fc1_in = functools.reduce(operator.mul, c2_out, 1)  # product of all elements in tuple
        self.fc1 = nn.Linear(fc1_in, 128)
        self.fc2 = nn.Linear(128, self.latent_dim)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.activation_function(c1)
        c2 = self.conv2(c1)
        c2 = self.activation_function(c2)
        f = flatten(c2)
        f1 = self.fc1(f)
        f1 = self.activation_function(f1)
        f2 = self.fc2(f1)
        out = self.activation_function(f2)
        return out


class ConvDecoder(nn.Module):

    def __init__(self, in_dim=20, out_shape=(1, 1, 1), activation_functino=F.relu, output_activation=torch.sigmoid):
        super(ConvDecoder, self).__init__()
        self.activation_function = activation_functino
        self.output_activation = output_activation
        self.out_shape = out_shape
        self.out_dim = functools.reduce(operator.mul, self.out_shape, 1)
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, self.out_dim)

    def forward(self, z):
        f1 = self.fc1(z)
        f1= self.activation_function(f1)
        f2 = self.fc2(f1)
        f2 = self.activation_function(f2)
        f3 = self.fc3(f2)
        out = self.output_activation(f3)
        out = out.view((out.shape[0],) + self.out_shape)  # batch size + shape of one sample
        return out


class bVAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, latent_dim=20, beta=1., activation_function=None,
                 optimizer=optim.Adam, recon_loss=F.mse_loss):
        super(bVAE, self).__init__()
        self.encoder = encoder if encoder is not None else ConvEncoder()
        self.decoder = decoder if decoder is not None else ConvDecoder()
        self.latent_dim = latent_dim
        self.beta = beta
        self.mu = nn.Linear(encoder.latent_dim, self.latent_dim)
        self.log_var = nn.Linear(encoder.latent_dim, self.latent_dim)
        self.activation_function = activation_function
        self.optimizer = optimizer(params=self.parameters(), weight_decay=1)
        self.reconstruction_loss = recon_loss
        self.epochs_trained = 0
        self.to(torch.double)


    def sample(self, mu, log_var):
        dim = self.latent_dim
        eps = torch.rand(dim).to(torch.double)
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
        mu_ = self.mu(x_)
        log_var_ = self.log_var(x_)

        if self.activation_function is not None:
            mu_ = self.activation_function(mu_)
            log_var_ = self.activation_function(log_var_)

        return mu_, log_var_

    def decode(self, z):
        return self.decoder.forward(z)

    def bvae_loss(self, y_pred, y, z_mu, z_var):
        recon_loss = self.reconstruction_loss(y_pred, y, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)
        loss = recon_loss + self.beta*kl_loss*0
        return loss

    def fit(self, data, labels, n_epochs=100, batch_size=128):
        history = []
        if type(data) is not torch.Tensor:
            data = torch.tensor(data)
            labels = torch.tensor(labels)
        set_size = data.shape[0]
        for n in range(n_epochs):
            print('STARTING EPOCH {}'.format(n))
            epoch_loss = []
            for i in range(0, set_size, batch_size):
                self.print_progress(((i+batch_size)/set_size)*100,
                                    progress='[{}/{}]'.format(min(i+batch_size,set_size), set_size))
                x, y = data[i:i+batch_size], labels[i:i+batch_size]
                self.optimizer.zero_grad()
                # forward
                y_pred, (z_mu, z_var) = self.forward(x)
                # evaluate
                loss = self.bvae_loss(y_pred, y, z_mu, z_var)
                epoch_loss.append(loss)
                # backprop
                loss.backward()
                self.optimizer.step()
                self.print_progress(((i+batch_size)/set_size)*100,
                                    progress='[{}/{}][loss {}]'.format(min(i+batch_size,set_size), set_size,
                                                                       loss/batch_size))
                pass
            self.epochs_trained += 1
            history.append(torch.mean(torch.tensor(epoch_loss))/set_size)
            print('\nFINISHED EPOCH {}, avg loss: {}\n\n'.format(n, history[-1]))
        return history

    def predict(self, x, mu_and_var=False):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        if mu_and_var:
            out = self.forward(x)
        else:
            out, _ = self.forward(x)
        return torch.tensor(out)

    def print_progress(self, x, progr_sym='.', progress='', scale=0.25):
        x = int(x*scale)
        p = '['+progr_sym*int(x+1)+' '*int(scale*100-x)+']'+progress
        print(p, end='\r', flush=True)


def save_model(model, path='./', modelname='model'):
    fullpath = os.path.join(path, modelname+'_state.tp')
    print('saving model to {}'.format(fullpath))
    if not os.path.exists(path):
        os.mkdir(path)
    checkpoint = {
        'model': model,
        'epoch': model.epochs_trained,
        'state_dict': model.state_dict(),
        'optimizer': model.optimizer.state_dict()
    }
    torch.save(checkpoint, fullpath)

def load_model(path, modelname, inference_only=False):
    fullpath = os.path.join(path, modelname+'_state'+'.tp')
    state = torch.load(fullpath)
    print('loaded checkpoint from {}'.format(fullpath))
    print('loading model...')
    model = state['model']
    model.load_state_dict(state['state_dict'])
    print('parameterize optimizer...')
    model.optimizer.load_state_dict(state['optimizer'])
    print('loading done')
    return model


if __name__ == '__main__':
    print("Testing bVAE")
    print("import")

    from torchvision.datasets import MNIST
    import numpy as np

    print("loading MNIST")

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = './data/test/'

    train = MNIST(root=root_dir, download=True, train=True)
    test = MNIST(root=root_dir, download=True, train=False)
    # num_train_samples = len(train)
    num_train_samples = 10000
    x_train, y_train = train.train_data[:num_train_samples], train.train_labels[:num_train_samples]
    x_test, y_test = test.test_data, test.test_labels
    x_train = np.expand_dims(x_train, 1) / 255
    x_test = np.expand_dims(x_test, 1) / 255
    data_shape = tuple(x_train.shape[1:])

    print("build encoder/ decoder")
    latent_dim = 32
    encoder = ConvEncoder(in_shape=data_shape, out_dim=64)  # out_dim == dim of mu and dim of log_var
    decoder = ConvDecoder(in_dim=latent_dim, out_shape=data_shape, )
    print(encoder.extra_repr())
    print(decoder.extra_repr())

    print("build beta vae")
    l = F.binary_cross_entropy
    # l = F.hinge_embedding_loss
    bvae = bVAE(encoder, decoder, latent_dim=latent_dim, recon_loss=l)
    print(bvae.extra_repr())
    print("fit bvae")
    history = bvae.fit(x_train, x_train, n_epochs=100, batch_size=128)
    print("saving model")
    path = './models_test'
    modelname = 'bvae_test'
    save_model(bvae, path=path, modelname=modelname)
    del bvae
    print("test loading")
    bvae = load_model(path=path, modelname=modelname)
    print("test bvae")
    test_out = bvae.predict(x_test[:25])


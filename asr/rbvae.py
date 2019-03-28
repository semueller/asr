import sys
import operator
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.utils import _pair
from asr.util import save_model, load_model

import numpy as np

from torch.nn import RNN, GRU, LSTM
import math


class RbVAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, latent_dim=20, beta=1., activation_function=None,
                 optimizer=optim.Adam, recon_loss=F.mse_loss):
        super(RbVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta = beta
        self.mu = nn.Linear(encoder.latent_dim, self.latent_dim)
        self.log_var = nn.Linear(encoder.latent_dim, self.latent_dim)
        self.activation_function = activation_function
        self.optimizer = optimizer(params=self.parameters(), weight_decay=1)
        self.reconstruction_loss = recon_loss
        self.epochs_trained = 0
        # self.to(torch.double)
        if self.activation_function is not None:
                print('bvae activation function set to {}'.format(self.activation_function))
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        mu_, log_var_, h_n = self.encode(x)
        seq_len = x.shape[1]
        z = self.sample(mu_, log_var_)
        out = self.decode(z, seq_len)
        return out, (mu_, log_var_)

    def encode(self, x):
        # convert x to flat vector
        embedding, h_n = self.encoder.forward(x)
        # calc mus and log_var from input
        mu_ = self.mu(embedding)
        log_var_ = self.log_var(embedding)
        if self.activation_function is not None:
                log_var_ = self.activation_function(log_var_)
                mu_ = self.activation_function(mu_)

        return mu_, log_var_, h_n

    def embed(self, x):
        mu_, log_var_, _ = self.encode(x)
        return mu_ + torch.exp(log_var_ /2)  # a 'sample' without epsilon

    def decode(self, z, seq_len):
        return self.decoder.forward(z, seq_len)

    def sample(self, mu, log_var):
        eps = torch.randn_like(log_var)  # rand_like samples uniformly from [0,1), randn_like samples from normal dist
        # exp(log_var * 1/2) -> exp(log(sqrt(var))) -> sqrt(var) -> stddev
        return mu + torch.mul(eps, torch.exp(log_var / 2))

    def bvae_loss(self, y_pred, y, z_mu, z_log_var):
        recon_loss = self.reconstruction_loss(y_pred, y, reduction='sum')
        # Paper: dkl = 0.5 * sum ( 1+ log(stddev**2) - mu**2 - stddev**2)
        dkl_loss = - 0.5 * torch.sum( 1. + z_log_var - z_mu**2 - torch.exp(z_log_var))
        loss = recon_loss + self.beta*dkl_loss
        return loss

    def fit(self, data, labels, validate=0.0, n_epochs=100, batch_size=128, converging_threshold=-1., path='./', periodic_save=0):
        history = []
        if type(data) is not torch.Tensor:
            data = torch.tensor(data)
            labels = torch.tensor(labels)
        set_size = data.shape[0]
        best_perform = math.inf
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

                self.print_progress(x=((i+batch_size)/set_size)*100,
                                    progress='[{}/{}][loss {}]'.format(min(i+batch_size,set_size), set_size,
                                                                       loss/batch_size))

            self.epochs_trained += 1
            history.append(torch.mean(torch.tensor(epoch_loss))/set_size)
            loss_delta = history[-2] - history[-1] if len(history)> 1 else -1

            if validate > 0:
                num_validation_samples = int(len(data)*validate)  # floors
                idxs = np.random.randint(0, len(data), num_validation_samples)
                x, y = data[idxs], labels[idxs]
                print('\n evaluate on {} samples; '.format(len(idxs)), end='')
                performance = self.evaluate(x, y)
                print('avg loss {}'.format(performance))
                if performance < best_perform:
                    best_perform = performance
                    save_model(self, path, '{}'.format(self.__class__.__name__))
            if periodic_save > 0:
                if self.epochs_trained % periodic_save == 0:
                    print('saving model')
                    save_model(self, path, '{}'.format(self.__class__.__name__))


            if 0 <= loss_delta and loss_delta <= converging_threshold:
                print('\n RETURN AFTER {} EPOCHS with loss_delta: {} < {} '.format(n,loss_delta, converging_threshold))
                break
            print('\nFINISHED EPOCH {}, avg loss: {}\n\n'.format(n, history[-1]))
        return history



    def evaluate(self, x, y):
        y_pred, (z_mu, z_var) = self.forward(x)
        # evaluate
        loss = self.bvae_loss(y_pred, y, z_mu, z_var)/len(x)
        return loss

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

if __name__=='__main__':
    import pickle as pkl

    from asr.encoder import GRUEncoder, LSTMEncoder
    from asr.decoder import GRUDecoder, LSTMDecoder

    data = pkl.load(open('/home/bing/sdb/testsets/mfccs_small.pkl', 'rb'))
    # print(data)
    X = torch.tensor(data['X'], dtype=torch.float)
    seqlen, nfeatures = X.shape[1:]

    latent_dim = 100
    hidden_size = 100
    encoder = LSTMEncoder(input_size=nfeatures, hidden_size=hidden_size, out_dim=int(latent_dim * 1.5))
    decoder = LSTMDecoder(input_size=latent_dim, hidden_size=hidden_size, out_dim=nfeatures)
    rbvae = RbVAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim, beta=1.1)
    print('num params: {}'.format(rbvae.num_params))
    rbvae.fit(X, X, batch_size=2, path='./models/', validate=0.0) # don't validate, does duplicate data on gpu, inefficient and throws out of memory exception

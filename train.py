import sys
import click
import pickle as pkl

import numpy as np

import torch
from asr.rbvae import RbVAE, RecDecoder, RecEncoder
from asr.util import save_model, load_model


@click.command()
@click.option('--data', type=str, default='./path/to/data.pkl', help='expects path to pickle containing a dict with the word as key')
@click.option('--model', type=str, default='./save/model/to/', help='expects path to pickle containing a dict with the word as key')
def main(data, model):

    print('loading data from {}'.format(data))
    filename = data.split('/')[-1].split('.')[0]
    data = pkl.load(open(data, 'rb'))
    print(data.keys())


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    train_percentage = 0.8
    x = data['X']
    seqlen, nfeatures = x.shape[1:]
    idxs = [i for i in range(len(x))]

    np.random.shuffle(idxs)
    split = np.math.ceil(len(idxs)*train_percentage)
    idxs_train, idxs_test =  idxs[:split], idxs[split:]
    with open(filename+'_idxs.dat', 'w') as f:
        f.write('train: \n {} \n test: \n {} \n'.format(idxs_train, idxs_test))

    x_train = torch.tensor(x, dtype=torch.float)
    del x

    latent_dim = 100
    hidden_size = 500
    encoder = RecEncoder(input_size=nfeatures, hidden_size=hidden_size, out_dim=int(latent_dim*1.5))
    decoder = RecDecoder(input_size=latent_dim, hidden_size=hidden_size, out_dim=nfeatures)
    rbvae = RbVAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim, beta=1.1)

    print('num params: {}'.format(rbvae.num_params))

    if device.type == 'cuda':
        x_train = x_train.to(device)
        rbvae = rbvae.to(device)
    history = rbvae.fit(x_train, x_train, batch_size=256, validate=0.0) # don't validate for now, duplicates data on gpu which raises memory exception (and is inefficient)
    modelname = '_'.join([rbvae.__class__.__name__, filename, str(latent_dim), str(hidden_size)])
    save_model(rbvae, path=model, modelname=modelname)
    pass

if __name__ == '__main__':
    main()
    sys.exit(0)

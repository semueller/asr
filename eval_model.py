import pickle as pkl
from asr.util import load_model

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP

import matplotlib.pyplot as plt

if __name__=='__main__':
    modelpath = '/home/bing/sdb/models'
    modelname = 'RbVAE_mfccs_100_500_100'
    datapath = '/home/bing/sdb/testsets/mfccs_50spc.pkl'

    data = pkl.load(open(datapath, 'rb'))

    data['X'] = torch.tensor(data['X'], dtype=torch.float)  # jibbles..
    model = load_model(modelpath, modelname)

    def embed(model, x):
        mu_, log_var_, h_n = model.encode(x)
        return mu_ + torch.exp(log_var_ / 2)  # a sample without epsilon


    print('compute encoding')
    embedding = embed(model, data['X'])
    embedding = embedding.detach().numpy()
    def calc_dist_matric(X):
        return euclidean_distances(X)
    print('computing {} distances'.format(len(embedding) ** 2))
    distances = calc_dist_matric(embedding)
    print(distances)
    f1 = plt.figure()
    plt.imshow(distances)

    print('calc dr with umap')
    um = UMAP()
    dr_embedding = um.fit_transform(embedding)

    print('plot')
    classes = np.unique(data['Y'])
    num_classes = len(classes)
    codebook = {c: n for c, n in zip(classes, range(num_classes))}
    coded = [codebook[i] for i in data['Y']]
    cmap = plt.get_cmap()
    colors = [cmap(1.*i/num_classes) for i in range(num_classes)]
    colors = [colors[i] for i in coded]
    f2 = plt.figure()
    plt.scatter(
        embedding[:, 0], embedding[:, 1], c=colors, cmap="Spectral", s=10
    )
    plt.show()

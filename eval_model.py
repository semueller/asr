import pickle as pkl
from asr.util import load_model

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP

import matplotlib.pyplot as plt
import pickle as pkl

if __name__=='__main__':
    modelpath = '/home/bing/sdb/models'
    modelname = 'RbVAE_fbank_26_100_500_2000'
    datapath = '/home/bing/sdb/testsets/fbank_26_50spc.pkl'
    dataset_name = datapath.split('/')[-1].split('.')[0]


    show = True
    save = True

    data = pkl.load(open(datapath, 'rb'))

    data['X'] = torch.tensor(data['X'], dtype=torch.float)  # jibbles..
    model = load_model(modelpath, modelname)

    print('compute encoding')
    embedding = model.embed(data['X'])
    embedding = embedding.detach().numpy()

    print('computing {} distances'.format(len(embedding) ** 2))
    distances = euclidean_distances(embedding)
    f1 = plt.figure()
    plt.imshow(distances)

    if save:
        plt.savefig('distances_{}.png'.format(dataset_name))
        pkl.dump(distances, open('distances_{}.pkl'.format(dataset_name), 'wb'))

    print('calc dimensionality reduction with umap')
    um = UMAP()
    dr_embedding = um.fit_transform(embedding)

    classes = np.unique(data['Y'])
    num_classes = len(classes)
    codebook = {c: n for c, n in zip(classes, range(num_classes))}
    coded = [codebook[i] for i in data['Y']]
    cmap = plt.get_cmap()
    colors = [cmap(1.*i/num_classes) for i in range(num_classes)]
    colors = [colors[i] for i in coded]
    f2 = plt.figure()
    plt.scatter(
        dr_embedding[:, 0], dr_embedding[:, 1], c=colors, cmap="Spectral", s=10
    )
    if save:
        plt.savefig(f'umap_scatter_{dataset_name}')
        pkl.dump(dr_embedding, open(f'umap_embedding_{dataset_name}.pkl', 'wb'))
    if show:
        plt.show()

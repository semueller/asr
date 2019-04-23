import pickle as pkl
from asr.util import load_model, Dataset, test_classifier, get_filenames

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP

import matplotlib.pyplot as plt
import pickle as pkl

import os

if __name__=='__main__':
    modelpath = '/home/bing/sdb/models/classification/mfccs'
    modelnames = get_filenames(modelpath, substr='tensor')
    datapath = '/home/bing/sdb/testsets/mfccs_50spc.pkl'
    dataset_name = 'mfccs_50spc'


    show = False
    save = True
    data: Dataset = pkl.load(open(datapath, 'rb'))

    data.data = torch.tensor(data.data, dtype=torch.float)  # jibbles..
    Y_cat = torch.tensor(data.get_labels_categorical(), dtype=torch.float32)

    for modelname in modelnames:
        # if not '_500_' in modelname:
        #     continue
        print(f'\n\n\nevaluating {modelname}')
        model = load_model(modelpath, modelname, dev='cpu')

        print('compute encoding')
        y_pred, h_t = model.forward(data.data)

        print(f'test on {dataset_name}')
        error_rate = test_classifier(model.forward, data.data, Y_cat)
        print(f'error on {dataset_name}: {error_rate}')
        y_c = torch.argmax(y_pred, 1)
        h_t = h_t.squeeze(0)
        embedding = h_t.detach().numpy()

        print('computing {} distances'.format(len(embedding) ** 2))
        distances = euclidean_distances(embedding)

        f = plt.figure(frameon=False)
        f.suptitle(modelname)
        f.set_size_inches(9, 7)
        # plt.subplot(2, 1, 1)
        plt.imshow(distances)
        plt.title(f'error on set: {error_rate*100:.2f}%')
        plt.colorbar()

        classes = data.classes
        num_classes = data.num_classes
        spc = 50
        for i in range(num_classes):
            plt.annotate(classes[i], [-90, i*50+25])

        if save:
            plt.savefig(f'plots/{modelname}_eval_dist.pdf')
            del f

        print('calc dimensionality reduction with umap')
        um = UMAP()
        dr_embedding = um.fit_transform(embedding)

        codebook = {c: n for c, n in zip(classes, range(num_classes))}
        coded =  [data.codebook[i] for i in data.labels]
        cmap = plt.get_cmap('nipy_spectral')
        colors = [cmap(1.*i/num_classes) for i in range(num_classes)]
        colors = [colors[i] for i in coded]
        # plt.subplot(2, 1, 2)
        f2 = plt.figure()
        for i in range(num_classes):
            plt.scatter(
                dr_embedding[i*50:i*50+50, 0], dr_embedding[i*50:i*50+50, 1], c=colors[i*50:i*50+50], cmap="nipy_spectral", s=10,
                label=classes[i]
            )
        plt.legend()
        if save:
            plt.savefig(f'plots/{modelname}_eval_umap.pdf')
            del f2
    del data
    if show:
        plt.show()
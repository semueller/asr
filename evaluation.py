import click
import pickle as pkl
from asr.util import load_model, Dataset, test_classifier, get_filenames, load_pkl, gmlvq_covered_variance
from sklearn_lvq import GmlvqModel

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP

import matplotlib.pyplot as plt
import pickle as pkl

import os

def eval_classifier(data, modelnames, destination, verbose):
    show = verbose
    save = True

    data.data = torch.tensor(data.data, dtype=torch.float)  # jibbles..
    Y_cat = torch.tensor(data.get_labels_categorical(), dtype=torch.float32)

    for modelname in modelnames:
        name = modelname.split('/')[-1]
        # if not '_500_' in modelname:
        #     continue
        print(f'\n\n\nevaluating {name}')
        # modelname is full path
        model = load_model(path=None, modelname=modelname, dev='cpu')

        print('compute encoding')
        y_pred, h_t = model.forward(data.data)

        error_rate = test_classifier(model.forward, data.data, Y_cat)
        print(f'error rate: {error_rate}')
        y_c = torch.argmax(y_pred, 1)
        h_t = h_t.squeeze(0)
        embedding = h_t.detach().numpy()
        del h_t, y_c

        print(f'computing {len(embedding**2)} distances')
        distances = euclidean_distances(embedding)
        del embedding
        f = plt.figure(frameon=False)
        f.suptitle(name)
        f.set_size_inches(9, 7)

        ax = f.add_subplot(111)
        plt.imshow(distances)
        del distances
        plt.title(f'error on set: {error_rate*100:.2f}%')
        plt.colorbar()

        classes = data.classes
        num_classes = data.num_classes
        spc = 50
        plt.xticks(np.arange(25, 1750, 50))
        plt.yticks(np.arange(25, 1750, 50))
        ax.set_xticklabels(classes, rotation=90)
        ax.set_yticklabels(classes)
        #for i in range(num_classes):

            #plt.annotate(classes[i], [-90, i*50+25])
        if save:
            plt.savefig(f'{destination}/{name}_eval_dist.pdf')
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

def eval_gmlvq(data, modelnames, destination, verbose):
    del data

    for model in modelnames:
        m: GmlvqModel = load_pkl(model)
        name = model.split('/')[-1].split('_')[0]
        thresh = 1
        dims, var, v, eig = gmlvq_covered_variance(m, thresh=thresh)
        f = plt.figure()
        ax1 = f.add_subplot(221)
        ax1.title.set_text(f'{name}\n#eigv explaining more \nthan {thresh}% variance = {dims}')
        ax1.bar(np.arange(len(eig)), eig, width=.5)
        ax1.set_xticks(range(len(eig)), range(len(eig)))
        # ax1.set_aspect()
        e = m.misc['test_error']
        print(f'{model}, {dims}, {e:.3f}')
        # path = f'{destination}/{name}.pdf'
        # plt.savefig(path)

        lmbd = m.omega_.conj().T.dot(m.omega_)
        for i in range(len(lmbd)):
            lmbd[i, i] = 0
        ax2 = f.add_subplot(222)
        # ax2.title.set_text(f'{name} Lambda Matrix')
        im = ax2.imshow(lmbd)
        f.colorbar(im, ax=ax2)
        ax2.title.set_text(f'Lambda')
        path = f'{destination}/{name}.pdf'
        plt.savefig(path, bbox_inches='tight')


    # plot eigenvalue spectrum


@click.command()
@click.option('--datapath', type=str, default='./path/to/data.pkl', help='expects path to Dataset containing embeddings')
@click.option('--modelpath', type=str, default='./path/to/ave/models/in', help='where to save trained models')
@click.option('--destination', type=str, default='./eval', help='path to destination folder of evaluation resutls')
@click.option('--idxs', type=str, default='./train_test_idxs.pkl',
              help='dict with train test keys containing list of indices')
@click.option('--modeltype', type=str)
def main(datapath, modelpath, destination, idxs, modeltype):
    verbose = False
    if modeltype == 'gmlvq':
        evalmethod = eval_gmlvq
    elif modeltype == 'classifier':
        evalmethod = eval_classifier
    else:
        raise ValueError(f'--modeltype {modeltype} not recognized')
        return -1

    if not os.path.exists(destination):
        os.system(f'mkdir -p {destination}')
        if not os.path.exists(destination):
            raise IOError(f'could not create destination path {destination}')

    modelnames = [os.path.join(modelpath, x) for x in get_filenames(modelpath)]
    data: Dataset = load_pkl(datapath)
    evalmethod(data, modelnames, destination, verbose=verbose)
    return 0


if __name__ == '__main__':
    e = main()
    exit(e)
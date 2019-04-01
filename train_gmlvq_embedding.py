import click

from sklearn_lvq import GmlvqModel

import pickle as pkl
import numpy as np

from asr.util import to_categorical


@click.command()
@click.option('--datapath', type=str, default='./path/to/data.pkl', help='expects path to pickle containing a dict with the word as key')
@click.option('--modelpath', type=str, default='./path/to/folder/with/models', help='expects path to pickle containing a dict with the word as key')
def main(modelpath, datapath, dataset_name='mfccs.pkl'):
    data = pkl.load(open(datapath, 'rb'))

    X = data['X'].detach().numpy()
    Y = np.argmax(to_categorical(data['Y']), 1)
    hiddim = datapath.split('/')[-1].split('_')[2]
    print(hiddim)
    gmlvq = GmlvqModel()
    print('train gmlvq')
    gmlvq.fit(X, Y)
    print('done, compute egenvalue decomposition of gmlvq.omega**2')
    pkl.dump(gmlvq, open(modelpath+f'/gmlvq{hiddim}.pkl', 'wb'))
    v, u = np.linalg.eig(gmlvq.omega_.conj().T.dot(gmlvq.omega_))
    print(v, u)
    pass

if __name__=='__main__':
    main()
    exit(0)

import click

from sklearn_lvq import GmlvqModel

import pickle as pkl
import numpy as np

from asr.util import gmlvq_covered_variance, Dataset, load_pkl, get_filenames, Dataset

def get_error(lvq_model: GmlvqModel, x, y) -> float:
    y_ = lvq_model.predict(x)
    errors = 1 - np.mean(y_ == y)
    return errors

@click.command()
@click.option('--datapath', type=str, default='./path/to/data.pkl', help='expects path to Dataset containing embeddings')
@click.option('--modelpath', type=str, default='./path/to/ave/models/in', help='where to save trained models')
@click.option('--idxs', type=str, default='./train_test_idxs.pkl',
              help='dict with train test keys containing list of indices')
def main(datapath, modelpath, idxs):
    dataset_names = get_filenames(datapath)
    print(f'using sets {dataset_names} from {datapath}')
    print('looking for idx files')
    idxs = load_pkl(idxs)
    idxs_train = idxs['train']
    idxs_test = idxs['test']

    for dataset_name in dataset_names:
        dataset: Dataset = load_pkl(dataset_name)
        X = dataset.data.detach().numpy()
        Y = dataset.labels.get_labels_numerical()
        x_train, x_test = X[idxs_train], X[idxs_test]
        y_train, y_test = Y[idxs_train], Y[idxs_test]
        hiddim = dataset_name.split('/')[-1].split('_')[2]
        print(f'training gmlvq on {hiddim} dim embedding')
        gmlvq = GmlvqModel()
        gmlvq.fit(x_train, y_train)
        train_error = get_error(gmlvq, x_train, y_train)
        test_error = get_error(gmlvq, x_test, y_test)
        var = gmlvq_covered_variance(gmlvq, thresh=1, verbose=True)
        misc = {'train_error': train_error, 'test_error' : test_error, 'matrix_var' : var}
        print(f'adding misc data to gmlvq model {misc}')
        gmlvq.misc = misc
        modelname = f'gmlvq{hiddim}.pkl'
        print(f'saving model to {modelname}')
        pkl.dump(gmlvq, open(modelpath+modelname, 'wb'))



if __name__=='__main__':
    main()
    exit(0)

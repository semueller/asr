import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc
from scipy.io import wavfile as wav
from scipy.spatial.distance import cdist
import os
import sys
import torch

# data = np.random.randint(0, 100, 16000)
# data = np.array([data, data+1, data+2, data+3, data+4])[:, :, np.newaxis]


def save_model(model, path='./', modelname='model'):
    fullpath = os.path.join(path, modelname+'_state.tp')
    print('saving model to {}'.format(fullpath))
    if not os.path.exists(path):
        os.mkdir(path)
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
    }
    if hasattr(model, 'epochs_trained'):
        checkpoint['epoch'] = model.epochs_trained

    if hasattr(model, 'optimizer'):
        checkpoint['optimizer'] = model.optimizer.state_dict()

    torch.save(checkpoint, fullpath)

def load_model(path, modelname, inference_only=False, dev='gpu'):
    if dev == 'gpu' and not torch.cuda.is_available():
        dev = 'cpu'
    fullpath = os.path.join(path, modelname+'_state'+'.tp') if '_state' not in modelname else os.path.join(path, modelname)
    print('loading checkpoint from {}'.format(fullpath))
    state = torch.load(fullpath, map_location=dev)
    print('loading model...')
    model = state['model']
    model.load_state_dict(state['state_dict'])
    print('parameterize optimizer...')
    if 'optimizer' in state.keys() and hasattr(model, 'optimizer'):
        model.optimizer.load_state_dict(state['optimizer'])
    print('loading done')
    if inference_only:
        print('Setting model into evaluation mode')
        model.eval()
    return model

def check_for_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    return device

def compute_distance_matrix():
    if not os.path.exists('/dev/shm/semueller/asr/npy/data_mfccs.npy'):
        SAMPLING_RATE = 16000  # from README.md
        data_mfccs = []
        pth = '/dev/shm/semueller/asr/npy/train_data.npy'
        #pth = './learn.npy'
        print('loading data')
        data = np.load(pth)
        print('calc mfccs')
        for x in data:
            x_m = mfcc(x, SAMPLING_RATE)
            data_mfccs.append(x_m)

        data_mfccs = np.array(data_mfccs)
        np.save('/dev/shm/semueller/asr/npy/data_mfccs', data_mfccs)
        sys.exit(-42)
    else:
        print('load mfccs')
    #    data_mfccs = np.load('./mfccs/data_mfccs.npy')
        data_mfccs = np.load('./mfccs/data_mfccs_out_50.npy')
    d = np.zeros(tuple([data_mfccs.shape[0]]*2))
    samples = data_mfccs.shape[0]
    size_d = samples**2
    print('calc distances')
    for i in range(samples):
        s_i = data_mfccs[i]
        print('{} of {}'.format(i*2, size_d))
        for j in range(i): # samples
            s_j = data_mfccs[j]
            res = np.sum([cdist(w_i[np.newaxis, :], w_j[np.newaxis, :]) for w_i, w_j in zip(s_i, s_j)])

            d[i, j] = res
            d[j, i] = res
    print('saving')
    np.save('/dev/shm/semueller/asr/distances', d)
    print('done')


def to_ranges(Y):
    raise NotImplementedError()


def get_subset(X, Y, samples_per_class=50, labelranges=None):
    if labelranges is None:
        labelranges = to_ranges(Y)

    if type(Y) != np.ndarray:
        Y = np.array(Y)

    sub_x, sub_y = [], []

    for c, start, end in labelranges:
        idxs = np.random.randint(start, end, samples_per_class)
        sub_x.append(X[idxs])
        sub_y.append(Y[idxs])

    sub_x = np.concatenate(tuple(sub_x), 0)
    sub_y = np.concatenate(tuple(sub_y), 0)

    return {
        'X': sub_x,
        'Y': sub_y
            }


def to_categorical(Y):
    class_labels = np.unique(Y)
    num_classes = len(class_labels)
    e = np.eye(num_classes)
    codebook = {l: i for i, l in enumerate(class_labels)}
    categories = np.array([e[codebook[l]] for l in Y])
    return categories

def error_rate(pred, target):
    _p = torch.argmax(pred, 1)
    _t = torch.argmax(target, 1)
    error = 1-torch.mean((_p == _t).to(torch.float32))
    return error

def test_classifier(classify, x, y, batch_size=256):
    with torch.no_grad():
        errors = []
        for i in range(0, len(x), batch_size):
            x_, y_ = x[i:i+batch_size], y[i:i+batch_size]
            y_p, _ = classify(x_)
            errors.append(error_rate(y_p, y_))
    return torch.mean(torch.tensor(errors))

# if __name__ == '__main__':
    # import pickle as pkl
    # print('test')
    # d = pkl.load(open('/home/bing/sdb/mfccs.pkl', 'rb'))
    # d_sub = get_subset(d['X'], d['Y'], labelranges=d['labelranges'])
    # pass
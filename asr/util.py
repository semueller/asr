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
        'epoch': model.epochs_trained,
        'state_dict': model.state_dict(),
        'optimizer': model.optimizer.state_dict()
    }
    torch.save(checkpoint, fullpath)

def load_model(path, modelname, inference_only=False, dev='gpu'):
    if dev == 'gpu' and not torch.cuda.is_available():
        dev = 'cpu'
    fullpath = os.path.join(path, modelname+'_state'+'.tp')
    print('loading checkpoint from {}'.format(fullpath))
    state = torch.load(fullpath, map_location=dev)
    print('loading model...')
    model = state['model']
    model.load_state_dict(state['state_dict'])
    print('parameterize optimizer...')
    model.optimizer.load_state_dict(state['optimizer'])
    print('loading done')
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

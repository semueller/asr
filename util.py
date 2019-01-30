import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc
from scipy.io import wavfile as wav
from scipy.spatial.distance import cdist
import os


# data = np.random.randint(0, 100, 16000)
# data = np.array([data, data+1, data+2, data+3, data+4])[:, :, np.newaxis]

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

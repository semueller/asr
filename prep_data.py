import numpy as np
from scipy.io import wavfile as wav
import os
import sys

import pickle as pkl
from asr.util import Dataset

from python_speech_features import mfcc, fbank, logfbank

def merge_set(path):
    prefix = '/'.join(path[:-1])
    target_dir = '/npy'
    if not os.path.exists(prefix+target_dir):
        os.mkdir(prefix+target_dir)

    dataset = path[-1]
    # words = ['one',
    #          'two',
    #          'three',
    #          'tree',
    #          'nine'
    #          ]
    words = [f for f in os.listdir(os.path.join(prefix, dataset)) if not os.path.isfile(os.path.join(prefix, dataset, f))]
    paths = ['/'.join([prefix, dataset, word]) for word in words]

    # path = paths[0]

    def convert(l):
        l = l.reshape(l.shape[0], 1)
        l = np.expand_dims(l, 0)
        return l

    files = []
    for path in paths:
        files.append([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    files = {w: f for w, f in zip(words, files)}
    for word in words:
        print(word)
        offset = 0
        target_l, l = 16000, 0
        while target_l != l:
            w = list(wav.read(files[word][offset])[1])
            offset += 1
            l = len(w)

        d_x = np.array(w)
        d_y = np.array([np.int16(0)]+w[:-1])
        d_x = convert(d_x)
        d_y = convert(d_y)
        # print(ones.shape)

        n = 0
        for f in files[word][offset:]:
            if len(d_x) > 5:
                break
            w = list(wav.read(f)[1])
            if len(w) != target_l:  # ones.shape[1]:
                n += 1
                continue
            w_x = np.array(w)
            w_y = np.array([np.int16(0)] + w[:-1])
            w_x = convert(w_x)
            w_y = convert(w_y)
            d_x = np.concatenate((d_x, w_x), axis=0)
            d_y = np.concatenate((d_y, w_y), axis=0)



        np.save(prefix+target_dir+'/'+word, d_x)
        np.save(prefix+target_dir+'/label_'+word, d_y)
        # np.save('./npy/l_'+str(target_l)+word+'pad1_y', ones_l)
        print('n {}'.format(n))
        print(d_x.shape)
        # print(ones[0].shape)
        # ones = ones.reshape(ones.sha [1] for f in files['one']])
    pass

def build_subset(path_npy = None, label_idxs = None, samples_per_class = 100, output_filename=None):
    assert path_npy is not None
    assert label_idxs is not None

    if output_filename is None:
        filename = path_npy.split('/')[-1].split('.')[0]
        output_filename = filename + '_out_{}'.format(samples_per_class)

    data = np.load(path_npy)
    print('data.shape: {}'.format(data.shape))

    res = np.empty((0,)+data.shape[1:])

    for idx in label_idxs:
        d = data[idx:idx+samples_per_class]
        print('range {}:{}, d shape {}'.format(idx, idx+samples_per_class, d.shape))
        res = np.concatenate((res,d), 0)

    print('output size {}'.format(res.shape))
    np.save('/dev/shm/semueller/asr/npy/'+output_filename, res)


def mfcc_to_dataset(mfcc):
    dic = pkl.load(open(pth, 'rb'))
    data = None
    labels = []
    labelranges = []
    for k, v in dic.items():
        v = np.array(v)
        v = normalize(v)
        data = np.concatenate((data, v), 0) if data is not None else v
        labels.extend([k]*len(v))
        labelranges.append((k, len(labels)-len(v), len(labels)-1))
    assert len(data) == len(labels)
    return {'X': data,
            'Y': labels,
            'labelranges': labelranges}

def normalize(x):
    # normalizes such that each filter (ie each dim per feature vector) is normal distributed along time axis
    for i in range(len(x)):
        x[i] = (x[i] - np.mean(x[i]))/np.std(x[i])
    return x

def fbank_to_datalabel(pth):
    dic = pkl.load(open(pth, 'rb'))
    data = None
    labels = []
    labelranges = []
    for k, v in dic.items():
        v = np.array([x[0] for x in v])
        v = normalize(v)
        data = np.concatenate((data, v), 0) if data is not None else v
        labels.extend([k]*len(v))
        labelranges.append((k, len(labels)-len(v), len(labels)-1))
    assert len(data) == len(labels)
    return {'X': data,
            'Y': labels,
            'labelranges': labelranges}

def zip_to_mfcc(pth):
    print('loading zipfile')
    data = np.load(pth, 'r')
    x = None
    labels = []
    labelranges = []
    for k in data.files[1:]:
        v = data[k]
        l = k.split('/')[-1]
        v = np.array(v)
        v = normalize(v)
        x = np.concatenate((x, v), 0) if x is not None else v
        labels.extend([l]*len(v))
        labelranges.append((l, len(labels)-len(v), len(labels)-1))
    print('{} datapoints loaded'.format(x.shape[0]))
    mfccs = np.zeros(shape=(x.shape[0], 99, 13))
    print('computing mfcc features')
    for i, timeseries in enumerate(x):
        mfccs[i] = mfcc(timeseries)

    print('done. \nreduced {}x{} dimensional timeseries to {}x{} dimensional features for {} samples'.format(
        x.shape[1], x.shape[2], mfccs.shape[1], mfccs.shape[2], mfccs.shape[0]
    ))

    dataset = Dataset(
        data=mfccs,
        labels=labels,
        labelranges=labelranges
    )

    return dataset

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) > 1:
        path = argv[1].split('/')
    else:
        path = './data/tf'.split('/')

    # merge_set(path)
    dataset = zip_to_mfcc('./data/npy.zip')
    pth = './data/mfccs.pkl'
    print('saving dataset to {}'.format(pth))
    pkl.dump(dataset, open(pth, 'wb'))
    subset = dataset.get_n_samples_per_class(10)
    subset_pth = './data/mfccs_50.pkl'
    print('saving subset to {}'.format(subset_pth))
    pkl.dump(subset, open(subset_pth, 'wb'))
    #
    #
    # with open('./mfccs/labelranges.txt','r') as file:
    #     lines = file.readlines()
    #
    # idxs = [0]
    # running_idx = 0
    # for line in lines[1:-1]:
    #     print(line)
    #     line = line.split(',')
    #     label = line[0]
    #     num_samples = int(line[1])
    #     running_idx += num_samples
    #     idxs.append(running_idx)
    #
    # build_subset('/dev/shm/semueller/asr/npy/train_label.npy', idxs, samples_per_class=60)

    # path[-1] = 'npy'
    # path = '/'.join(path)
    # print(path)
    # files = [
    #     'fbank', 'fbank_26',
    #     'mfccs', 'logfbank'
    # ]
    # format = 2
    # merged = mfcc_to_dataset(path.format(files[format])) if files[format] == 'mfccs' else fbank_to_datalabel(path.format(files[format]))
    # print(merged.keys(), merged['labelranges'])
    # with open('/home/bing/sdb/{}.pkl'.format(files[format]), 'wb') as f:
    #     pkl.dump(merged, f)
    sys.exit(0)
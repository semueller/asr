import numpy as np
from scipy.io import wavfile as wav
import os
import sys

def merge_set(argv = None):
    if len(argv) > 1:
        path = argv[1].split('/')
    else:
        path = './data/tf'.split('/')
    prefix = '/'.join(path[:-1])
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


        np.save(prefix+'/npy/'+word, d_x)
        np.save(prefix+'/npy/label_'+word, d_y)
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
        output_filename = filename + 'out'

    data = np.load(path_npy)
    print('data.shape: {}'.format(data.shape))

    res = np.empty((1,)+data.shape[1:])

    for idx in label_idxs:
        print('range {}:{}'.format(idx, idx+samples_per_class))
        res = np.concatenate((res,data[idx:idx+samples_per_class]), 0)

    np.save(output_filename, res)


if __name__ == '__main__':
    # merge_set(sys.argv)


    with open('./mfccs/labelranges.txt','r') as file:
        lines = file.readlines()

    idxs = [0]
    running_idx = 0
    for line in lines[1:]:
        print(line)
        line = line.split(',')
        label = line[0]
        num_samples = int(line[1])
        running_idx += num_samples
        idxs.append(running_idx)

    build_subset('/dev/shm/semueller/asr/npy/data_mfccs.npy', idxs)

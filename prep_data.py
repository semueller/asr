import numpy as np
from scipy.io import wavfile as wav
import os
import sys

if __name__ == '__main__':

    if len(sys.argv) > 1:
        path = sys.argv[1].split('/')
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
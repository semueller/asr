import numpy as np
from scipy.io import wavfile as wav
import os
import pyaudio as paud
import wave

def play_wav_files(files):
    CHUNK = 1024

    p = paud.PyAudio()
    for f in files:
        wf = wave.open(f, 'rb')
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(CHUNK)

        while data != '':
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()

    p.terminate()

if __name__ == '__main__':

    prefix = './data'
    dataset = 'tf'
    # words = ['one',
    #          'two',
    #          'three',
    #          'tree',
    #          'nine'
    #          ]
    words = [f for f in os.listdir(os.path.join(prefix, dataset)) if not os.path.isfile(os.path.join(prefix, dataset, f))]
    paths = ['/'.join([prefix, dataset, word]) for word in words]

    # path = paths[0]

    files = []
    for path in paths:
        files.append([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    files = {w: f for w, f in zip(words, files)}
    for word in words:
        print(word)
        offset = 0
        target_l, l = 16000, 0
        while target_l != l:
            w = wav.read(files[word][offset])
            w = list(wav.read(files[word][offset])[1])
            offset += 1
            l = len(w)

        ones = np.array(w)
        # ones_l = np.array([0]+w[:-1])
        ones = ones.reshape(ones.shape[0], 1)
        # ones_l = ones_l.reshape(ones_l.shape[0], 1)
        ones = np.expand_dims(ones, 0)
        # ones_l = np.expand_dims(ones_l, 0)
        # print(ones.shape)

        n = 0
        for f in files[word][offset:]:
            w = wav.read(f)
            if len(w[1]) != target_l:#ones.shape[1]:
                n+=1
                continue
            w = w[1]
            # w = list(w[1])
            # w_l = np.array([0]+w[:-1])
            # w = np.array(w)
            # w = w.reshape(len(w), 1)
            # w_l = w_l.reshape(w.shape)
            w = np.expand_dims(w, 0)
            # w_l = np.expand_dims(np.array(w_l), 0)
            ones = np.concatenate((ones, w), axis=0)
            # ones_l = np.concatenate((ones_l, w_l), axis=0)


        np.save('./npy/'+word, ones)
        # np.save('./npy/l_'+str(target_l)+word+'pad1_y', ones_l)
        print('n {}'.format(n))
        print(ones.shape)
        # print(ones[0].shape)
        # ones = ones.reshape(ones.sha [1] for f in files['one']])
    pass
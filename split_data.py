import pickle as pkl
import numpy as np

data = pkl.load(open('./mfccs.pkl', 'rb'))

lr = data.labelranges
train = 0.9
test = 1-train

idxs_train, idxs_test = [], []

for c, s, e in lr:
    n_samples = e-s  # starts at 0
    n_train = int(n_samples*train)
    idxs = np.arange(s, e+1)
    np.random.shuffle(idxs)
    idxs_train.extend(idxs[:n_train])
    idxs_test.extend(idxs[n_train:])

pkl.dump(idxs_train, open('idxs_train.pkl', 'wb'))
pkl.dump(idxs_test, open('idxs_test.pkl', 'wb'))

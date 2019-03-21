from asr.rnn import RecurrentNetwork
from asr.cells import *

import torch
import random
import math

import matplotlib.pyplot as plt
try:
    import seaborn
    seaborn.set()
except:
    pass

# generate a time series for the tutorial task
def generate_sine_cosine(T = 100, tau = None, phaseshift = 0.):
    '''grabbed this from Benjamin Paassen [bpaassen@techfak.uni-bielefeld.de]'''
    # first, select a wavelength at random in the range [20, 50]
    if(tau is None):
        tau = random.randrange(20, 51)
    # generate the times at which we sample the waves
    times = torch.Tensor(list(range(0, T)))
    # generate the input signal, i.e. the cosine
    X = torch.cos((times / tau + phaseshift) * 2. * math.pi)
    # generate the output signal, i.e. the sine
    Y = torch.sin((times / tau + phaseshift) * 2. * math.pi)
    # Set the first quarter-wavelength to zero
    Y[:int(tau / 4.)] = 0
    # unsqueeze the second axes to have standard form data
    X = X.unsqueeze(1)
    Y = Y.unsqueeze(1)
    return X, Y, times

if __name__=='__main__':

    celltype = 'rnn'
    model = RecurrentNetwork(celltype=celltype, in_dim=1, hidden_dim=10, out_dim=1)

    (X, Y, times) = generate_sine_cosine()
    # f1 = plt.figure(figsize=(14, 6))
    # plt.plot(list(times), list(X), list(times), list(Y))
    # plt.xlabel('t')
    # plt.ylabel('amplitude')
    # plt.legend(['X', 'Y'])
    # plt.show()
    loss_threshold = 1E-2
    learning_curve = []

    num_train_sets = 500
    data = []
    for _ in range(num_train_sets):
        data.append(generate_sine_cosine())
    # apply the model with the current parameters
    X = [x[0] for x in data]
    Y = [y[1] for y in data]
    Ts = [times[2] for times in data]
    learning_curve = model.fit(X, Y)
    print(learning_curve)

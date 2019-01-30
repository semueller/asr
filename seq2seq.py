import os
import numpy as np
import pickle as pkl

import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

latent_dim = 20
timesteps = 16000
input_dim = 1


def done():
    print('done')

def load():


    path = '/dev/shm/semueller/asr/npy/'

    if os.path.exists(path+'train_data.npy') and os.path.exists(path+'train_label.npy'):
        train_data = np.load(path+'train_data.npy')
        train_label = np.load(path+'train_label.npy')
    else:
        dpath = path+'data'
        lpath = path+'label'
        words = [f for f in os.listdir(dpath)]
        labels = [f for f in os.listdir(lpath)]
        data = {f.split('.')[0]: np.load(os.path.join(dpath, f)).astype(np.float64) for f in words}
        labels = {f.split('_')[1].split('.')[0]: np.load(os.path.join(lpath, f)).astype(np.float64) for f in labels}


        # normalize
        for k, _ in data.items():
            d = data[k]
            d -= np.mean(d)
            d /= np.max(d)
            l = labels[k]
            l[:, 1:, :] = d[:, :-1, :]

        s = data['forward'].shape
        train_data, train_label = np.empty((1,) + s[1:]), np.empty((1,) + s[1:])
        for k, _ in data.items():
            v = data[k]
            l = labels[k]
            train_data = np.concatenate((train_data, v), 0)
            train_label = np.concatenate((train_label, l),0)

        np.save(path+'train_data', train_data)
        np.save(path+'train_label', train_label)

    return train_data, train_label

print('loading data')
train_data, train_label = load()
done()
train_perc = .8
test_perc = 1 - train_perc
print('shuffling data')
idx_dict = {'train_perc': train_perc}
num_samples = train_data.shape[0]
idxs = np.random.permutation(range(num_samples))
num_train_samples = np.floor(num_samples*train_perc).astype(np.int64)
idxs_train = idxs[:num_train_samples]
idxs_test = idxs[num_train_samples:]
idx_dict['train'] = idxs_train
idx_dict['test'] = idxs_test
test_data = train_data[idxs_test]
test_label = train_label[idxs_test]
train_data = train_data[idxs_train]
train_label = train_label[idxs_train]
done()

print('build model')
# build model
try:
    sess = tf.Session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=20))
    tf.keras.backend.set_session = sess
except Exception:

    pass


save = True

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

sequence_autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                             metrics=['accuracy'])
sequence_autoencoder.summary()

h = None

filepath="/dev/shm/semueller/asr/checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

h = sequence_autoencoder.fit(x=train_data, y=train_label,
                         # validation_data=(test_data, test_label),
                         batch_size=64, epochs=100, verbose=1, callbacks=callbacks_list)


if save:
    file_base_path = "/dev/shm/models/"
    ae_json = sequence_autoencoder.to_json()
    model_name = "sequence_autoencoder"
    if not os.path.exists(file_base_path+model_name):
        os.makedirs(file_base_path)

    with open(file_base_path+model_name+".json", 'w') as file:
        file.write(ae_json)
        sequence_autoencoder.save_weights(file_base_path+model_name+".weights")

    with open(file_base_path+'history.pkl', 'wb') as hfile:
        pkl.dump(h, hfile)

    with open(file_base_path+'train_test_idxs.pkl', 'wb') as idxfile:
        pkl.dump(idx_dict, idxfile)

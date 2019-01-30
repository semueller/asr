# lstm autoencoder recreate sequence
from numpy import array
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model

# define input sequence
# sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10, 11, 12, 13])
# reshape input into [samples, timesteps, features]
# n_in = len(sequence)
# data = sequence.reshape((1, n_in, 1))

def load():
    return np.load('./ones.npy'), np.load('./ones_l.npy')

# define model
data, data_dev = load()

input_seq = Input(shape=(None, 1))#,1))
l1 = LSTM(100, activation='relu', return_state=True)
enc_out, h, c = l1(input_seq)
enc_state = [h, c]
# recu = RepeatVector(16000)(enc_out)
latent_in = Input(shape=(None, 1))
dec = LSTM(100, activation='relu', return_sequences=True)
dec_out, _, _ = dec(latent_in, initial_state=enc_state)

# out = TimeDistributed(Dense(1))(dec)
# out = TimeDistributed()(dec_out)


# encoder = Model(input_seq, l1)
# decoder = Model(latent_in, out)
# ae_out = decoder(encoder(input_seq))
ae_2 = Model([input_seq, latent_in], dec_out)
# ae = Model(input_seq, ae_out)
ae_2.summary()
ae_2.compile(optimizer='adam', loss='mse')
# fit model
plot_model(ae_2, show_shapes=True, to_file='lstm_encoder.png')
ae_2.fit([data, data[:, 1:, ]], data, epochs=300, verbose=0)
# connect the encoder LSTM as the output layer
# plot_model(encoder, show_shapes=True, to_file='lstm_encoder.png')
# get the feature vector for the input sequence
# yhat = encoder.predict(sequence)
# print(yhat.shape)
# print(yhat)
# lstm autoencoder recreate sequence
from numpy import array
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model

def load():
    return np.load('./ones.npy')

# define input sequence
# reshape input into [samples, timesteps, features]
data = load()
n_in = 0
# data = np.expand_dims(data, 0)
# define model
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(None, None, 1)))
# model.add(RepeatVector(16000))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
model.summary()
# fit model
model.fit(data, data, epochs=300, verbose=1)
# connect the encoder LSTM as the output layer
model = Model(inputs=model.inputs, outputs=model.layers[0].output)
plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
# get the feature vector for the input sequence
# yhat = model.predict(data)
# print(yhat.shape)
# print(yhat)
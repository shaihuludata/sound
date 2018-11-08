import matplotlib.pylab as P
from keras.models import Model, Sequential
from keras.layers import Conv1D, Input, Activation
import numpy as np

# x_train = P.randn(100000)
# y_train = 2*x_train
# x_val = P.randn(10000)
# y_val = 2*x_val

fir_filter = np.array([0.1, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.1], dtype='float32')
signal_v = np.array([1.0, 2.0, 3.0, 8.0, 4.0, 2.0, 3.0, 4.0, 6.0, 2.0, 4.0, 6.0], dtype='float32')
signal_n = np.array([1.0, 2.5, 3.0, 8.0, 4.0, 2.5, 3.0, 4.0, 6.5, 2.0, 4.5, 6.0], dtype='float32')
batch_size = signal_n.shape[0]
nSampleSize = signal_n.shape[0]

myinput = Input(shape=(None, 1))  # shape = (BATCH_SIZE, 1D signal)
output = Conv1D(1, # output dimension is 1
                # fir_filter.shape[0], # filter length is 15
                15,
                padding="valid")(myinput)

model = Sequential()  # inputs=myinput, outputs=output)
model.add(Conv1D(filters=8, kernel_size=12, strides=3, padding='valid', use_bias=False, input_shape=(12, 1), name='c1d', activation='relu'))
model.add(Activation('relu', input_shape=(nSampleSize, 1)))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])

# model.fit(x_train, y_train, batch_size=batch_size, epochs=100, shuffle=False, validation_data=(x_val, y_val))

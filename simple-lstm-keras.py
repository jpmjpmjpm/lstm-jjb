# Testing https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


if __name__ == '__main__':
    sequence = np.linspace(10, 300, num=30)
    n_steps = 3
    X, y = split_sequence(sequence, n_steps)

    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    model.fit(X, y, epochs=200, verbose=0)

    x_test = np.array([340, 350, 360])
    x_test = x_test.reshape((1, n_steps, n_features))
    ypred = model.predict(x_test, verbose=1)
    print(ypred)

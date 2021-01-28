import mxnet as mx
import mxnet.gluon.nn as nn
import mxnet.gluon.rnn as rnn
import numpy as np
from mxnet.autograd import record


def split_sequence(sequence: np.array, n_steps: int) -> tuple:
    inputs, targets = [], []

    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        inputs.append(seq_x)
        targets.append(seq_y)

    # MXNet LSTM wants to have (seq. length, batch size, feature size) for dimensions of inputs
    # inputs = np.transpose(np.array(inputs).reshape(27, 3, 1), (1, 0, 2))
    inputs = np.array(inputs).reshape(27, 3, 1)
    targets = np.array(targets).reshape(-1, 1)

    return inputs, targets


if __name__ == '__main__':
    nb_epochs = 200
    sequence = np.linspace(10, 300, num=30, dtype='float32')
    n_steps = 3
    n_features = 1
    learning_rate = 0.01
    xnp, ynp = split_sequence(sequence, n_steps)

    x = mx.ndarray.from_numpy(xnp, zero_copy=False)
    y = mx.ndarray.from_numpy(ynp, zero_copy=False)

    network = nn.Sequential()
    network.add(rnn.LSTM(50, 1))
    network.add(nn.Dense(1))
    network.initialize(mx.init.Xavier())

    criterion = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(network.collect_params(), optimizer='adam', optimizer_params={'learning_rate': 0.03})

    for epoch in range(nb_epochs):
        with mx.autograd.record():
            out = network(x)
            loss = criterion(out, y)

        loss.backward()
        trainer.step(batch_size=x.shape[1])

    x_test = np.array([340, 350, 360])
    x_test = x_test.reshape((n_steps, 1, n_features))

    x_test = mx.ndarray.from_numpy(x_test)
    ypred = network(x_test)
    print(ypred)

import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch import zeros, FloatTensor, no_grad


def create_inout_sequences(input_data, tw):
    inout_seq = []
    input_data_count = len(input_data)
    for i in range(input_data_count - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_layer_size: int = 50, nb_layers: int = 1,
                 output_size: int = 1) -> None:
        super(SimpleLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.nb_layers = nb_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, nb_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (zeros(1, 1, self.hidden_layer_size),
                            zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def main():
    # Define the sequence of points to be modeled
    train_data = np.linspace(10, 300, num=30, dtype='float32')

    # Normalize the data (list of lists with a single element)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

    # Split the sequence by n_steps length slices and transform it to tensors
    n_steps = 4
    train_data_normalized = FloatTensor(train_data_normalized).view(-1)
    train_inout_seq = create_inout_sequences(train_data_normalized, n_steps)
    print(f"Train in/out sequence: {train_inout_seq}")

    nb_epochs = 200
    learning_rate = 0.01
    model = SimpleLSTM()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    print(model)
    single_loss = None
    for i in range(nb_epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (zeros(1, 1, model.hidden_layer_size),
                                 zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1 and single_loss is not None:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    if single_loss is not None:
        print(f'epoch: {nb_epochs:3} loss: {single_loss.item():10.10f}')

    model.eval()
    test_x_list = [150 + 10 * i for i in range(n_steps)]
    test_x = np.array(test_x_list, dtype='float32')
    test_x_normalized = scaler.transform(test_x.reshape(-1, 1))
    test_x_normalized = FloatTensor(test_x_normalized).view(-1)

    with no_grad():
        model.hidden = (zeros(1, 1, model.hidden_layer_size),
                        zeros(1, 1, model.hidden_layer_size))
        test_y_normalized = model(test_x_normalized)
    test_y = test_y_normalized.numpy()
    test_y = scaler.inverse_transform(test_y.reshape(-1, 1))

    print(f"Predicted value for {test_x_list} sample: {test_y[0][0]:.2f}")


if __name__ == '__main__':
    main()

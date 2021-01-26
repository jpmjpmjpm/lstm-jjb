import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def split_sequence(sequence: np.array, n_steps: int) -> tuple:
    inputs, targets = [], []

    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        inputs.append(seq_x)
        targets.append(seq_y)

    return np.array(inputs), np.array(targets)


class SimpleLSTMwithCells(nn.Module):
    def __init__(self):
        super(SimpleLSTMwithCells, self).__init__()

        self.lstm1 = nn.LSTMCell(1, 50)
        self.lstm2 = nn.LSTMCell(1, 50)
        self.lstm3 = nn.LSTMCell(1, 50)

        self.fc = nn.Linear(50, 1)

    def forward(self, input):
        h0, c0 = torch.zeros(input.size(0), 50, dtype=torch.float), torch.zeros(input.size(0), 50, dtype=torch.float)
        x0 = input[:, 0, :]
        x1 = input[:, 1, :]
        x2 = input[:, 2, :]
        h1, c1 = self.lstm1(x0, (h0, c0))
        h2, c2 = self.lstm2(x1, (h1, c1))
        h3, c3 = self.lstm3(x2, (h2, c2))
        out = self.fc(h3)
        return out


if __name__ == '__main__':
    torch.manual_seed(0)
    nb_epochs = 200
    sequence = np.linspace(10, 300, num=30, dtype='float32')
    n_steps = 3
    n_features = 1
    learning_rate = 0.01
    x, y = split_sequence(sequence, n_steps)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x = x.reshape(-1, n_steps, n_features)
    y = y.reshape(-1, n_features)

    model = SimpleLSTMwithCells()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(nb_epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    model.eval()
    x_test = torch.tensor([340, 350, 360], dtype=torch.float).reshape(-1, n_steps, n_features)
    ypred = model(x_test)
    print(ypred)

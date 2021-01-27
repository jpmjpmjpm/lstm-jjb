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


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_dim: int = 50, nb_layers: int = 1, output_size: int = 1) -> None:
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.nb_layers = nb_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, nb_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

        # Try different types of parameters initialization
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)
        # for name, param in self.lstm.named_parameters():
        #     if 'bias' in name:
        #         nn.init.zeros_(param)
        #     elif 'weight' in name:
        #         nn.init.xavier_uniform_(param)

    def forward(self, input):
        h0 = torch.randn(self.nb_layers, input.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.randn(self.nb_layers, input.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(input, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        # return torch.unsqueeze(out, 2)
        return out


if __name__ == '__main__':

    # Define the sequence of points to be modeled
    sequence = np.linspace(10, 300, num=30, dtype='float32')

    # Split the sequence by n_steps length slices -> x, to predict y
    n_steps = 3
    x, y = split_sequence(sequence, n_steps)
    print(f"Train seq (numpy): {x}")
    print(f"Train label (numpy): {y}")

    nb_epochs = 200
    n_features = 1
    learning_rate = 0.01
    model = SimpleLSTM()
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x = x.reshape(-1, n_steps, n_features)
    y = y.reshape(-1, n_features)
    print(f"Train seq (tensor): {x}")
    print(f"Train label (tensor): {y}")

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

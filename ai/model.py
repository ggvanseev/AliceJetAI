import torch
import torch.nn as nn
import numpy as np

# lstm-fc
class LSTM_FC(nn.Module):
    def __init__(
        self,
        input_size=4,
        hidden_size=[20, 4],
        output_dim=5,
        num_layers=2,
        batch_size=1,
        device=torch.device("cpu"),
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.output_dim = output_dim
        h0 = self.hidden_init().to(self.device)
        c0 = self.hidden_init().to(self.device)
        self.hidden = (h0, c0)

        print("************** Model ****************")
        print("Neural Network:\tLSTM")
        print("Input Size:\t%d" % self.input_size)
        print("hidden Size:\t[%d,%d]" % (self.hidden_size[0], self.hidden_size[1]))
        print(f"Ouput dim: \t{self.output_dim}")
        print("No. of Layers:\t%d" % self.num_layers)
        print("Batch Size:\t%d" % self.batch_size)
        print("*************************************")

        super(LSTM_FC, self).__init__()

        # lstm + 2FC
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size[0],
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.05,
        ).to(self.device)
        self.fc1 = nn.Linear(
            in_features=self.hidden_size[0], out_features=self.output_dim, bias=True
        ).to(self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=self.hidden_size[1], out_features=2, bias=True
        ).to(self.device)

    def forward(self, x):
        # lstm + 2FC
        x, hidden = self.lstm(x, self.hidden)
        x = [i[-1] for i in x]
        x = self.fc1(torch.stack(x, dim=0).to(self.device))
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # used for initialization of hidden states and cell states
    def hidden_init(self):
        # h_0, c_0
        # (num_layers, batch_size, hidden_size)
        return torch.autograd.Variable(
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size[0])
        )


class LSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2
    ):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)  # fully connected layer
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(self.device),
            weight.new(self.n_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(self.device),
        )
        return hidden


class SentimentNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        drop_prob=0.5,
    ):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )
        return hidden


# Optional, for later
class GRUNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2
    ):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        )
        return hidden

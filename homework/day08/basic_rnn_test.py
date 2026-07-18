import torch
from torch import nn
from torch.nn.functional import embedding

from homework.day07.advanced_cnn_test import criterion


class ModelOfRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(ModelOfRNNCell, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn_cell = torch.nn.RNNCell(input_size, hidden_size)

    def forward(self, input, hidden):
        return self.rnn_cell(input, hidden)

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


class ModelOfRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(ModelOfRNN, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


class ModelWithEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, embedding_size, num_class, num_layers=1):
        super(ModelWithEmbedding, self).__init__()
        self.emb = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        x = self.emb(x)
        x,_ = self.rnn(x, h)
        x = self.fc(x)
        return x.view(-1, self.num_class)

def test01():
    batch_size = 2
    seq_len = 3
    input_size = 4
    hidden_size = 2

    cell = torch.nn.RNNCell(input_size, hidden_size)

    # (seq,batch,features)
    dataset = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(batch_size, hidden_size)

    for idx, input in enumerate(dataset):
        print('-' * 20, idx, '=' * 20)
        print('Input size:', input.shape)

        hidden = cell(input, hidden)

        print('outputs size:', hidden.shape)
        print(hidden)


def test02():
    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2
    num_layers = 1

    cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    # (seqLen,batchSize,inputSize)
    inputs = torch.randn(batch_size, seq_len, input_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    out, hidden = cell(inputs, hidden)

    print('Output size:', out.shape)
    print('Output:', out)
    print('Hidden size:', hidden.shape)
    print('Hidden:', hidden)


def test03():
    batch_size = 1
    input_size = 4
    hidden_size = 4
    num_layers = 1
    seq_len = 5

    # 'hello' → 'ohlol'
    idx2char = ['e', 'h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]
    y_data = [3, 1, 2, 3, 2]

    one_hot_lookup = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    x_one_hot = [one_hot_lookup[x] for x in x_data]

    # rnn_cell
    # inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
    # labels = torch.LongTensor(y_data).view(-1, 1)

    # rnn
    inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
    labels = torch.LongTensor(y_data)

    # model = ModelOfRNNCell(input_size, hidden_size, batch_size)
    model = ModelOfRNN(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # rnn_cell_train(criterion, idx2char, inputs, labels, model, optimizer)
    rnn_train(criterion, idx2char, inputs, labels, model, optimizer)

def test04():
    num_class = 4
    input_size = 4
    hidden_size = 8
    embedding_size = 10
    num_layers = 2
    batch_size = 1
    seq_len = 5

    # 'hello' → 'ohlol'
    idx2char = ['e', 'h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]
    y_data = [3, 1, 2, 3, 2]

    inputs = torch.LongTensor(x_data).view(batch_size,seq_len)
    labels = torch.LongTensor(y_data)

    model = ModelWithEmbedding(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size, embedding_size=embedding_size, num_layers=num_layers,num_class=num_class)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(15):
        optimizer.zero_grad()
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        _,idx = y_pred.max(1)
        idx = idx.data.numpy()
        print('Predicted:', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))




def rnn_train(criterion, idx2char, inputs, labels, model, optimizer):
    for epoch in range(15):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(1)
        idx = idx.data.numpy()
        print('Predicted:', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


def rnn_cell_train(criterion, idx2char, inputs, labels, model, optimizer):
    for epoch in range(15):
        loss = 0
        optimizer.zero_grad()
        h = model.init_hidden()
        print('Predicted string:', end='')
        for input, label in zip(inputs, labels):
            h = model(input, h)
            loss += criterion(h, label)
            _, idx = h.max(1)
            print(idx2char[idx.item()], end='')
        loss.backward()
        optimizer.step()
        print(', Epoch [%d/15 loss=%.4f]' % (epoch + 1, loss.item()))


if __name__ == '__main__':
    test04()

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

path = './dataset/mnist'

train_set = datasets.MNIST(root=path, train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

test_set = datasets.MNIST(root=path, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)


class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=2)
        self.conv3 = torch.nn.Conv2d(20, 30, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(30 * 2 * 2, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(self.relu(self.conv1(x)))  # 1*28*28 26*26 13*13
        x = self.pooling(self.relu(self.conv2(x)))  # 12*12 6*6
        x = self.pooling(self.relu(self.conv3(x)))  # 4*4 2*2
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(10, 15, kernel_size=2)
        self.conv3 = torch.nn.Conv2d(15, 20, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(20 * 2 * 2, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(self.relu(self.conv1(x)))  # 1*28*28 26*26 13*13
        x = self.pooling(self.relu(self.conv2(x)))  # 12*12 6*6
        x = self.pooling(self.relu(self.conv3(x)))  # 4*4 2*2
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(10, 30, kernel_size=2)
        self.conv3 = torch.nn.Conv2d(30, 60, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(60 * 2 * 2, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(self.relu(self.conv1(x)))  # 1*28*28 26*26 13*13
        x = self.pooling(self.relu(self.conv2(x)))  # 12*12 6*6
        x = self.pooling(self.relu(self.conv3(x)))  # 4*4 2*2
        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


model1 = Net1()
model2 = Net2()
model3 = Net3()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.5)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.5)
optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01, momentum=0.5)

criterion = torch.nn.CrossEntropyLoss()


def train(epoch, model, optimizer):
    total_loss = 0.0
    running_loss = 0.0
    model.train()
    batch_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        running_loss += loss.item()
        batch_count += 1
        if batch_idx % 300 == 299:
            print('Train Epoch: {}, Loss: {:.6f}'.format(epoch + 1, running_loss / 300))
            running_loss = 0.0
    return total_loss / batch_count


def test(epoch, model):
    model.eval()
    correct = 0
    total_count = 0
    with torch.no_grad():
        for data, target in test_loader:
            y_pred = model(data)
            _, predicted = torch.max(y_pred.data, dim=1)
            total_count += target.size(0)
            correct += (predicted == target).sum().item()
    print('Epoch: {}, Accuracy: {:.2f}%'.format(epoch + 1, 100 * correct / total_count))
    return 100 * correct / total_count


def start(model, optimizer, model_name):
    print('{} start'.format(model_name))
    epoch_list = []
    loss_list = []
    acc_list = []
    for epoch in range(10):
        loss_list.append(train(epoch, model, optimizer))
        acc_list.append(test(epoch, model))
        epoch_list.append(epoch + 1)
    fig, ax1 = plt.subplots()
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_loss)
    line1 = ax1.plot(epoch_list, loss_list, color=color_loss, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()
    color_acc = 'tab:blue'
    ax2.set_ylabel('Accuracy(%)', color=color_acc)
    line2 = ax2.plot(epoch_list, acc_list, color=color_acc, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    ax1.set_title('Loss and Accuracy of {}'.format(model_name))
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    fig.tight_layout()
    plt.show()

    print('{} end Accuracy'.format(model_name, acc_list[-1]))


if __name__ == '__main__':
    start(model1, optimizer1, 'model1')
    start(model2, optimizer2, 'model2')
    start(model3, optimizer3, 'model3')

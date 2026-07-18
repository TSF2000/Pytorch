from pathlib import Path
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_ROOT = Path(__file__).resolve().parents[2] / 'dataset' / 'mnist'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST(MNIST_ROOT, train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

test_set = datasets.MNIST(MNIST_ROOT, train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1_avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch1_conv = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1, stride=1)

        self.branch2_conv = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1)

        self.branch3_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1)
        self.branch3_conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, stride=1, padding=2)

        self.branch4_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1)
        self.branch4_conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.branch4_conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch1 = self.branch1_conv(self.branch1_avg_pool(x))
        branch2 = self.branch2_conv(x)
        branch3 = self.branch3_conv2(self.branch3_conv1(x))
        branch4 = self.branch4_conv3(self.branch4_conv2(self.branch4_conv1(x)))

        return torch.cat((branch1, branch2, branch3, branch4), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=88, out_channels=20, kernel_size=5)

        self.inception1 = Inception(in_channels=10)
        self.inception2 = Inception(in_channels=20)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.inception1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.inception2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.rb1 = ResidualBlock(16)
        self.rb2 = ResidualBlock(32)
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rb1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rb2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


def train(epoch, model, optimizer, criterion):
    total_loss, running_loss = 0.0, 0.0
    batch_count = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(data)
        loss = criterion(y_hat, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        running_loss += loss.item()
        batch_count += 1
        if batch_count % 300 == 0:
            print('Train Epoch: {}, Batch Count: {}, Loss: {:.6f}'.format(epoch + 1, batch_count, running_loss / 300))
            running_loss = 0.0
    return total_loss / batch_count


def test(epoch, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            y_hat = model(data)
            total += target.size(0)
            _, predicted = torch.max(y_hat.data, 1)
            correct += predicted.eq(target).sum().item()
        print('Epoch: {}, Accuracy: {:.2f}%'.format(epoch + 1, 100 * correct / total))
        return 100 * correct / total


def train_and_test(epochs, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    epoch_list = []
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        epoch_list.append(epoch)
        loss_list.append(train(epoch, model, optimizer, criterion))
        acc_list.append(test(epoch, model))
    return epoch_list, loss_list, acc_list


if __name__ == '__main__':
    model1 = Net()
    model2 = Net2()
    epoch_list1, loss_list1, acc_list1 = train_and_test(epochs=10, model=model1)
    epoch_list2, loss_list2, acc_list2 = train_and_test(epochs=10, model=model2)

    plt.plot(epoch_list1, loss_list1, label='Loss of Model(Inception)')
    plt.plot(epoch_list2, loss_list2, label='Loss of Model(ResidualBlock)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epoch_list1, acc_list1, label='Accuracy % of Model(Inception)')
    plt.plot(epoch_list2, acc_list2, label='Accuracy % of Model(ResidualBlock)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

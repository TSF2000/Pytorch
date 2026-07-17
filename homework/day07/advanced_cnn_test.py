from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt

# homework/day07 → 项目根 / dataset/mnist
MNIST_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "mnist"

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(root=str(MNIST_ROOT), train=True, download=True, transform=train_transforms)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

test_set = datasets.MNIST(root=str(MNIST_ROOT), train=False, download=True, transform=train_transforms)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # branch1 avg_pool 1*1_24_conv
        self.branch1_conv = nn.Conv2d(in_channels, 24, kernel_size=1)

        conv_1_1_16 = nn.Conv2d(in_channels, 16, kernel_size=1)
        # branch2 1*1_16_conv
        self.branch2_conv = conv_1_1_16

        # branch3 1*1_16_conv 5*5_24_conv
        self.branch3_conv1 = conv_1_1_16
        self.branch3_conv2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # branch4 1*1_16_conv 3*3_24_conv 3*3_24_conv
        self.branch4_conv1 = conv_1_1_16
        self.branch4_conv2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch4_conv3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        pass

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, 3, 1, 1)
        branch1 = self.branch1_conv(avg_pool)

        branch2 = self.branch2_conv(avg_pool)

        branch3 = self.branch3_conv1(x)
        branch3 = self.branch3_conv2(branch3)

        branch4 = self.branch4_conv1(x)
        branch4 = self.branch4_conv2(branch4)
        branch4 = self.branch4_conv3(branch4)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(88, 20, 5)
        self.inception1 = InceptionA(in_channels=10)
        self.inception2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.inception1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.inception2(x)
        x = x.view(batch_size, -1)
        return self.fc(x)


model = Net()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def train(epoch):
    total_loss, running_loss, batch_cnt = 0.0, 0.0, 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
        batch_cnt += 1
        if batch_cnt % 300 == 299:
            print('epoch:{}, batch_idx:{}, avg_loss:{:.4f}'.format(epoch + 1, batch_idx, running_loss / 300))
            running_loss = 0.0
    return total_loss / batch_cnt


def test(epoch):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            _, predicted = torch.max(model(data), 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print(('epoch:{}, acc: {:.2f}%'.format(epoch + 1, 100 * correct / total)))
        return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    loss_list = []
    for epoch in range(10):
        loss_list.append(train(epoch))
        acc_list.append(test(epoch))
        epoch_list.append(epoch)
    plt.plot(epoch_list, acc_list, label='Accuracy')
    plt.plot(epoch_list, loss_list, label='Loss')
    plt.legend()
    plt.show()

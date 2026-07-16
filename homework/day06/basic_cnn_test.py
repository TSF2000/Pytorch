import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_set = datasets.MNIST('./dataset/mnist', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

test_set = datasets.MNIST('./dataset/mnist', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(self.relu(self.conv1(x)))  # conv1 batch,1,28,28 → batch,10,(28-5+1),(28-5+1) pooling batch,10,12,12
        x = self.pooling(self.relu(self.conv2(x)))  # conv2 batch,10,12,12 → batch,20,8,8   pooling batch,20,4,4
        x = x.view(batch_size, -1)  # 20 * 4 * 4 = 320
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = torch.nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    batch_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
        batch_count += 1
        if batch_count % 300 == 299:
            print('epoch:{}, avg_loss:{}'.format(epoch + 1, running_loss / batch_count))
            running_loss = 0.0
    return total_loss / batch_count


def test():
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            y_pred = model(data)
            _, predicted = torch.max(y_pred.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy:%d%%' % (100 * correct / total))
    return 100 * correct / total


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

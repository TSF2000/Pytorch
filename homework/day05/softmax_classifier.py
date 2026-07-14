import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        return self.l5(x)


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    running_loss = 0.0
    batch_count = 0
    total_loss = 0.0
    for batch_idx, (x_data, y_data) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        batch_count += 1
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
    return total_loss / batch_count


def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x_data, y_data) in test_loader:
            y_pred = model(x_data)
            _, predicted = torch.max(y_pred.data, dim=1)
            total += y_data.size(0)
            correct += (predicted == y_data).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100 * correct / total


if __name__ == '__main__':
    loss_list = []
    epoch_list = []
    acc_list = []
    for epoch in range(10):
        avg_loss = train(epoch)
        loss_list.append(avg_loss)
        epoch_list.append(epoch)
        acc = test()
        acc_list.append(acc)

    fig, ax1 = plt.subplots()
    color_loss = 'tab:blue'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color_loss)
    line1 = ax1.plot(epoch_list, loss_list, color=color_loss,label='train loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()
    color_acc = 'tab:red'
    ax2.set_ylabel('accuracy (%)', color=color_acc)
    line2 = ax2.plot(epoch_list, acc_list, color=color_acc,label='test accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    ax1.set_title("Training Loss & Test Accuracy")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels,loc='best')

    fig.tight_layout()
    plt.show()
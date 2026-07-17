from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt

# homework/day07 → 项目根 / dataset/mnist
MNIST_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "mnist"
# 权重保存在脚本同级 checkpoints/
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
CKPT_PATH = CKPT_DIR / "inception_mnist.pth"          # state_dict（推荐）
CKPT_FULL_PATH = CKPT_DIR / "inception_mnist_full.pth"  # 整个模型

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()


def train(epoch, model, optimizer):
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
            print('epoch:{}, batch_idx:{}, avg_loss:{:.4f}'.format(epoch + 1, batch_cnt + 1, running_loss / 300))
            running_loss = 0.0
    return total_loss / batch_cnt


def test(epoch, model):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            _, predicted = torch.max(model(data), 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print('epoch:{}, acc: {:.2f}%'.format(epoch + 1, 100 * correct / total))
        return correct / total


def save_checkpoint(model, path, epoch=None, acc=None, optimizer=None):
    """保存权重 state_dict；可选附带 epoch / acc / optimizer。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if epoch is not None:
        payload["epoch"] = epoch
    if acc is not None:
        payload["acc"] = acc
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    print("saved checkpoint ->", path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    """加载 state_dict 到已构建好的 model（结构须与训练时一致）。"""
    map_location = map_location or device
    payload = torch.load(path, map_location=map_location, weights_only=False)
    # 兼容：既支持本课的 dict，也支持纯 state_dict
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    print("loaded checkpoint <-", path)
    return payload


def save_full_model(model, path):
    """直接保存整个模型对象（含结构 + 权重）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 存 CPU 版更通用，避免换机器无 GPU 时加载失败
    torch.save(model.cpu(), path)
    model.to(device)  # 存完把模型放回原设备，方便继续训练
    print("saved full model ->", path)


def load_full_model(path, map_location=None):
    """
    加载整个模型。不必再 Net()，但运行环境里仍要能找到 InceptionA / Net 的类定义
    （通常就是在同一个 .py 里，或能 import 到定义这些类的模块）。
    """
    map_location = map_location or device
    model = torch.load(path, map_location=map_location, weights_only=False)
    model = model.to(device)
    model.eval()
    print("loaded full model <-", path)
    return model


def predict(model, x):
    """单张/小批量推理，x shape: (N,1,28,28)"""
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
    return pred


if __name__ == '__main__':
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    epoch_list = []
    acc_list = []
    loss_list = []
    best_acc = 0.0

    for epoch in range(10):
        loss_list.append(train(epoch, model, optimizer))
        acc = test(epoch, model)
        acc_list.append(acc)
        epoch_list.append(epoch)
        # 测试集准确率创新高时：存 state_dict + 整个模型
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, CKPT_PATH, epoch=epoch + 1, acc=best_acc, optimizer=optimizer)
            save_full_model(model, CKPT_FULL_PATH)

    print("best acc during training: {:.2f}%".format(100 * best_acc))

    # ----- 方式 A：state_dict（需先 Net() 再建结构）-----
    model_from_state = Net().to(device)
    ckpt = load_checkpoint(CKPT_PATH, model_from_state)
    print("state_dict meta: epoch={}, acc={:.2f}%".format(
        ckpt.get("epoch"), 100 * ckpt.get("acc", 0.0)))
    test(ckpt.get("epoch", 1) - 1, model_from_state)

    # ----- 方式 B：整个模型（直接 load，不必先 Net()）-----
    model_full = load_full_model(CKPT_FULL_PATH)
    test(0, model_full)

    # 随便抽一张测试图做预测（用整模加载的结果）
    sample, label = test_set[0]
    pred = predict(model_full, sample.unsqueeze(0))
    print("sample predict: {}, label: {}".format(pred.item(), label))

    plt.plot(epoch_list, acc_list, label='Accuracy')
    plt.plot(epoch_list, loss_list, label='Loss')
    plt.legend()
    plt.show()

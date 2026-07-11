import torch
import matplotlib.pyplot as plt

# 准备数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重
w = torch.Tensor([1.0])
w.requires_grad_(True)


# 前馈函数
def forward(x):
    return w * x


# 损失函数
def loss(x, y):
    return (forward(x) - y) ** 2

epoch_list = []
loss_list = []


print("predict (before training): )", 4, forward(4).item())

# 训练
for epoch in range(200):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()  # 自动计算梯度
        print("\tgrad:", x, y, w.grad.item())
        w.data = w.data - 0.001 * w.grad.data
        w.grad.data.zero_()
    epoch_list.append(epoch)
    loss_list.append(l.item())
    print("progress:", epoch, l.item())
print("predict(after training): ", 4, forward(4).item())

plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
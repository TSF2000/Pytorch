import torch
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

# 定义模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel()

# 优化器和损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss(reduction='sum')

w_list = []
b_list = []
loss_list = []
epoch_list = []

# 训练

for epoch in range(10000):
    # 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()

    # 反馈
    loss.backward()

    # 更新
    optimizer.step()

    epoch_list.append(epoch)
    w_list.append(model.linear.weight.item())
    b_list.append(model.linear.bias.item())
    loss_list.append(loss.item())

print("w=",model.linear.weight.item())
print("b=",model.linear.bias.item())
print("y_pred=", model(x_data).detach())

x_range = torch.linspace(0.5, 3.5, 100).reshape(-1, 1)
y_range = model(x_range).detach()

plt.figure()
plt.plot(x_range.numpy(), y_range.numpy(), label="sigmoid")
plt.scatter(x_data.numpy(), y_data.numpy(), color="red", label="data")
plt.xlabel("x")
plt.ylabel("probability")
plt.legend()
plt.show()

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200,1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability')
plt.grid()
plt.show()
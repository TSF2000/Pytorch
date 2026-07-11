import torch

# 准备数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])   # shape: (3, 1)
y_data = torch.Tensor([[2.0], [4.0], [6.0]])   # shape: (3, 1)

# 构建模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# 损失 + 优化器
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练1
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch:{}, loss:{}'.format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 查看参数
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

# 测试
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=',y_test.data)
import torch
import matplotlib.pyplot as plt

# 准备数据
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

# 定义模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinearModel()

# 定义优化器和损失函数
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
criterion = torch.nn.MSELoss(reduction='sum')

# 结果可视化准备
epoch_list = []
w_list = []
b_list = []
loss_list = []

# 训练（前馈、反馈、更新）
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print("epoch:{},loss:{}".format(epoch,loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_list.append(epoch)
    w_list.append(model.linear.weight.item())
    b_list.append(model.linear.bias.item())
    loss_list.append(loss.item())

# 查看参数
print("w=",model.linear.weight.item())
print("b=",model.linear.bias.item())

# 测试
x_test = torch.Tensor([[4.0],[5.0],[6.0]])
y_test = model(x_test)
print("y_test=",y_test)

# plt.plot(epoch_list,loss_list)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
plt.plot(epoch_list,w_list,label="w")
plt.plot(epoch_list,b_list,label="b")
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.show()

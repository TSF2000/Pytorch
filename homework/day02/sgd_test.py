import matplotlib.pyplot as plt

# stochastic gradient descent

# 1.准备数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 2.初始化权重
w = 1.0


# 3.forward函数
def forward(x):
    return w * x


# 4.cost函数 单样本
def cost(x, y):
    return (forward(x) - y) ** 2


# 5.梯度函数
def gradient(x, y):
    return 2 * x * (w * x - y)


# 6.记录训练过程
epoch_list = []
loss_list = []

print("before training", 4, forward(4))
# 7.循环训练
for epoch in range(100):
    for x,y in zip(x_data, y_data):
        cost_val = cost(x, y)
        grad_val = gradient(x, y)
        w -= 0.01 * grad_val
    epoch_list.append(epoch)
    loss_list.append(cost_val)
    print("epoch:", epoch, "w:", w, "loss:", cost_val)

print("after training", 4, forward(4))
# 8.结果可视化
plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()
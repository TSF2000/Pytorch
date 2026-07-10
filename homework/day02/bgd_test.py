import matplotlib.pyplot as plt

# mini-batch gradient descent

# 1.准备数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 2.初始化权重
w = 1.0


# 3.定义模型
def forward(x):
    return w * x


# 4.cost 函数(全数据集MSE)
def cost(xs, ys):
    cost_val = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost_val += (y_pred - y) ** 2
    return cost_val / len(xs)


# 5.定义梯度函数
def gradient(xs, ys):
    grad = 0
    # d/dw [(wx-y)^2] = 2x(wx-y)
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


# 6.记录训练过程，用于画图
epoch_list = []
cost_list = []

# 7.训练循环 BGD(batch gradient descent)
for epoch in range(200):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w = w - 0.01 * grad_val
    print("epoch:{}, w:{}, cost:{}, grad:{}".format(epoch, w, cost_val, grad_val))
    epoch_list.append(epoch)  # 横轴：轮次
    cost_list.append(cost_val)  # 纵轴：Loss

# 8.训练结果可视化
print('predict (after training)',4,forward(4))
plt.plot(epoch_list, cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()

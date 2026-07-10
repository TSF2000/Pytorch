import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


W = np.arange(0.0,4.1,0.1)
B = np.arange(-2.0,2.0,0.1)
w_gird,b_grid = np.meshgrid(W,B)
mse_gird = np.zeros_like(w_gird)

for i,w in enumerate(W):
    for j,b in enumerate(B):
        l_sum = 0
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val,y_val)
            l_sum += loss_val
        mse_gird[j,i] = l_sum/len(x_data)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(w_gird,b_grid,mse_gird)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('mse')
plt.show()
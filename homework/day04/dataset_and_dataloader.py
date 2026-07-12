import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


# 准备数据

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.from_numpy(data[:, [-1]])
        self.len = data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# 构建模型
class DiabetesModel(torch.nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

dataset = DiabetesDataset("../../courseware/diabetes.csv.gz")
train_loader = DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0)
model = DiabetesModel()

# 损失和优化器
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_list = []
epoch_list = []

# 训练
for epoch in range(200):
    total_loss = 0.0
    batch_count = 0
    for i,(inputs,targets) in enumerate(train_loader,0):
        y_pred = model(inputs)
        loss = criterion(y_pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
    avg_loss = total_loss / batch_count
    epoch_list.append(epoch)
    loss_list.append(avg_loss)

plt.plot(epoch_list,loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

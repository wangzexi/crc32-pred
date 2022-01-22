import torch
from torch.nn import functional as F
from torch import nn

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

# 自己创建一个数据集


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()

        dataset_x = torch.randint(
            low=0, high=2, size=(1000, 2), dtype=torch.int64)
        dataset_y = dataset_x[:, 0] ^ dataset_x[:, 1]
        dataset_y = dataset_y.reshape(dataset_y.shape[0], -1)

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

    def __len__(self):
        return self.dataset_x.shape[0]

    def __getitem__(self, idx):
        # 返回数据样本，由特征x，和目标y组成
        return (self.dataset_x[idx].float(), self.dataset_y[idx].float())


class MyModel(pl.core.lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # [b, 1]
        x = self.fc(x)  # [b, 1]
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self.forward(x)
        loss = F.mse_loss(y, pred_y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self.forward(x)
        loss = F.mse_loss(y, pred_y)

        pred_y = torch.where(pred_y > 0.5, torch.ones_like(
            y), torch.zeros_like(y))

        acc = torch.sum(pred_y == y).float() / y.shape[0]

        print(f'loss: {loss.item()}, acc: {acc.item()}')

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)  # 设置我们优化器


model = MyModel()  # 创建模型

train_dataset = MyDataset()  # 创建训练数据集
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# training
trainer = pl.Trainer(
    gpus=1,
    max_epochs=300,
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader
)

print(list(model.parameters()))  # 输出模型经过训练后，拟合到的w和b，应该接近于我们数据集中设定的2.6和4

# 测试
test_dataset = MyDataset()  # 创建训练数据集
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

trainer.test(
    model=model,
    test_dataloaders=test_dataloader
)

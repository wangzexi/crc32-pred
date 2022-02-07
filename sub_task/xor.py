import torch
from torch.nn import functional as F
from torch import nn

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

# 数据集


class XorDataset(Dataset):
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


class XorDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32):
        super().__init__()

        self.batch_size = batch_size

    def prepare_data(self):
        dataset = XorDataset()

        # train:val:test = 6:2:2
        self.train = torch.utils.data.Subset(
            dataset, range(0, int(len(dataset)*0.6)))
        self.valid = torch.utils.data.Subset(
            dataset, range(int(len(dataset)*0.6), int(len(dataset)*0.8)))
        self.test = torch.utils.data.Subset(
            dataset, range(int(len(dataset)*0.8), int(len(dataset))))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False)


class XorModel(pl.core.lightning.LightningModule):
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self.forward(x)
        loss = F.mse_loss(y, pred_y)

        pred_y = torch.where(pred_y > 0.5, torch.ones_like(
            y), torch.zeros_like(y))

        acc = torch.sum(pred_y == y).float() / y.shape[0]

        self.log_dict({'valid_loss': loss, 'valid_acc': acc})

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self.forward(x)
        loss = F.mse_loss(y, pred_y)

        pred_y = torch.where(pred_y > 0.5, torch.ones_like(
            y), torch.zeros_like(y))

        acc = torch.sum(pred_y == y).float() / y.shape[0]

        self.log_dict({'test_loss': loss, 'test_acc': acc})

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)  # 设置我们优化器


trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='XorModel'),
    max_epochs=200,

    callbacks=[pl.callbacks.ModelCheckpoint(  # 保存在验证集上最好的模型
        filename='best',
        monitor='valid_acc',
        mode='max',
    )]
)

datamodule = XorDataModule()  # 创建数据集

model = XorModel()  # 创建模型

# 训练
trainer.fit(
    model=model,
    datamodule=datamodule
)

# model = XorModel.load_from_checkpoint('sub_task/pre_training/xor.ckpt')
# print(list(model.parameters()))

# 测试
trainer.test(
    model=model,
    datamodule=datamodule
)

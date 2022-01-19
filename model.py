import torch
import torchmetrics
import pytorch_lightning as pl


class MyModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(32)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
        )

        self.mse_loss = torch.nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.Precision(),
            torchmetrics.Recall(),
            torchmetrics.F1(),
        ])
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='valid/')
        self.test_metrics = metrics.clone(prefix='test/')

    def forward(self, x):
        x = self.bn(x)
        x = self.mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        output = self.train_metrics(logits, y)
        output['train/loss'] = loss
        self.log_dict(output)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        output = self.valid_metrics(logits, y)
        output['valid/loss'] = loss
        self.log_dict(output)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        output = self.test_metrics(logits, y)
        output['test/loss'] = loss
        self.log_dict(output)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

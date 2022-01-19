import pytorch_lightning as pl
from train import MyDataModule, MyModel


model = MyModel()
model.load_from_checkpoint(
    'tb_logs/Model/version_4/checkpoints/best_F1.ckpt')

data_module = MyDataModule(data_path='./data/urandom.bin')

trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='Model')
)

trainer.test(
    model=model,
    datamodule=data_module,
)

import pytorch_lightning as pl

from model import MyModel
from dataset import MyDataMudule


trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='Model'),
    max_epochs=50,
    log_every_n_steps=50,

    callbacks=[pl.callbacks.ModelCheckpoint(  # 保存在验证集上最好的模型
        filename='best_F1',
        # filename='{valid/F1:.2f}-{epoch}-{step}',
        monitor='valid/F1',
        save_top_k=1,
        mode='max',
    )]
)

model = MyModel()
data_module = MyDataMudule(batch_size=512)

trainer.fit(
    model=model,
    datamodule=data_module,
)

trainer.test(  # 使用测试集测试模型
    model=model,
    datamodule=data_module,
)

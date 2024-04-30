from model import LocationBind
from dataloader import INatDataset
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import random
import numpy as np
import os


def seed_everything(seed=42):
    """
    seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__=='__main__':
    seed_everything()
    train_dataset = INatDataset('../metaformer', 'train.json')
    val_dataset = INatDataset('../metaformer', 'val.json')
    model = LocationBind(train_dataset, val_dataset)
    torch.cuda.empty_cache()
    logger = WandbLogger(project="Eco-Bind", name="location-bind")
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='locbind-{epoch:02d}-{val_loss:.2f}',
        mode='min'
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='ddp_find_unused_parameters_false',
        max_epochs=1500,
        num_nodes=1,
        callbacks=[checkpoint],
        accumulate_grad_batches=16,
        logger=logger,
        log_every_n_steps=1
        )
    trainer.fit(model)
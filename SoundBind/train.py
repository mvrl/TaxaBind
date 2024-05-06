from model import AudioBind
from dataloader import INatDataset
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import random
import numpy as np
import os
from config import cfg
from argparse import ArgumentParser

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

    parser = ArgumentParser(description='')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--devices', type=str, default="auto")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--run_name', type=str, default='test')

    args = parser.parse_args()

    seed_everything()
    train_dataset = INatDataset(data_file=cfg['train_df'])
    val_dataset = INatDataset(data_file=cfg['val_df'])
    kwargs = {'batch_size':args.batch_size, 'num_workers': args.num_workers}
    
    model = AudioBind(train_dataset, val_dataset, **kwargs)
    torch.cuda.empty_cache()

    ckpt_save_path = os.path.join(cfg['log_path'],"sound-bind",args.run_name)
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    print("Saving checkpoints to:",ckpt_save_path)

    logger = WandbLogger(save_dir=cfg['log_path'],project="Eco-Bind", name="sound-bind",mode=args.wandb_mode)
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=ckpt_save_path,
        filename='soundbind-{epoch:02d}-{val_loss:.2f}',
        mode='min',
        save_top_k=3
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.devices,
        strategy='ddp_find_unused_parameters_false',
        max_epochs=1500,
        num_nodes=1,
        callbacks=[checkpoint],
        accumulate_grad_batches=16,
        logger=logger,
        log_every_n_steps=1
        )
    trainer.fit(model)
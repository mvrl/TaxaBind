from easydict import EasyDict as edict

config = edict()
config.train_df = 'train_df.json'
config.val_df= 'val_df.csv'

config.batch_size = 256
config.lr = 1e-4
config.accumulate_grad_batches = 8
config.max_epochs = 20
config.num_workers = 16
config.devices = 2
config.val_check_interval = 0.5


config.save_dir = 'checkpoints'
config.filename = 'soundbind-{epoch:02d}-{val_loss:.2f}'

config.locked_tuning = True
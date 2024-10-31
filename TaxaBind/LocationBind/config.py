from easydict import EasyDict as edict

config = edict()
config.img_dir = '../taxabind_data/'
config.train_json_path = 'train.json'
config.val_json_path = 'val.json'

config.batch_size = 512
config.lr = 1e-4
config.accumulate_grad_batches = 8
config.max_epochs = 20
config.num_workers = 16
config.devices = 2
config.val_check_interval = 0.5


config.save_dir = 'checkpoints'
config.filename = 'locbind-{epoch:02d}-{val_loss:.2f}'

config.locked_tuning = True
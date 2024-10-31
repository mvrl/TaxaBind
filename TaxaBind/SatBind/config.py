from easydict import EasyDict as edict

config = edict()
config.img_dir = '../taxabind_data/'
config.imo_dir = '../taxabind_satellite/taxabind_sentinel/images/sentinel/'
config.imo_dir_val = '../taxabind_satellite/taxabind_val_sentinel/images/sentinel/'
config.train_json_path = '../taxabind_data/train.json'
config.val_json_path = '../taxabind_data/val.json'

config.batch_size = 256
config.lr = 1e-4
config.accumulate_grad_batches = 8
config.max_epochs = 20
config.num_workers = 16
config.devices = 2
config.val_check_interval = 0.5
config.sat_encoder = 'openai/clip-vit-base-patch16'

config.save_dir = 'checkpoints'
config.filename = 'satbind-{epoch:02d}-{val_loss:.2f}'

config.locked_tuning = True
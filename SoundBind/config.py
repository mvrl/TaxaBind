import os
cfg ={}

cfg['data_path'] = "/storage1/jacobsn/Active/proj_smart/inat_image_sounds"
cfg["train_df"] = "/storage1/jacobsn/Active/proj_smart/inat_image_sounds/train_df.csv"
cfg["val_df"] = "/storage1/jacobsn/Active/proj_smart/inat_image_sounds/validate_df.csv"
cfg["test_df"] = "/storage1/jacobsn/Active/proj_smart/inat_image_sounds/test_df.csv"
cfg['log_path'] = "/home/k.subash/EcoBind/logs"


def replace_storage_path(path):
    return path.replace("/storage1/", "/storage1/fs1/")

def check_and_replace_path(path):
    if not os.path.exists(path):
        return replace_storage_path(path)
    return path

cfg['data_path'] = check_and_replace_path(cfg['data_path'])
cfg['train_df'] = check_and_replace_path(cfg['train_df'])
cfg['val_df'] = check_and_replace_path(cfg['val_df'])
cfg['test_df'] = check_and_replace_path(cfg['test_df'])



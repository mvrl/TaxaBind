# For the final set of samples containing valid data, performs train/val/test split.

import pandas as pd
import os
import numpy as np

data_path = "/scratch/k.subash/data/inat_image_sounds/"

meta_df = pd.read_csv(os.path.join(data_path,"metadata.csv"))
failed_ids = list(pd.read_csv(os.path.join(data_path,"failed_ids.csv"))['id'])
final_df = meta_df[~meta_df['id'].isin(failed_ids)]
final_df = final_df.to_csv(os.path.join(data_path,"metadata_final.csv"))

meta_df = pd.read_csv(os.path.join(data_path,"metadata_final.csv"))

##Split IDs
train_df, validate_df, test_df = np.split(meta_df.sample(frac=1, random_state=42), [int(.85*len(meta_df)), int(.9*len(meta_df))])
print("sample count:train/val/test for ratio 85:5:10",(len(train_df),len(validate_df)),len(test_df)) #sample count:train/val/test for ratio 85:5:10 (74910, 4407) 8813
splits_path = data_path
train_df.to_csv(os.path.join(splits_path,'train_df.csv'))
validate_df.to_csv(os.path.join(splits_path,'validate_df.csv'))
test_df.to_csv(os.path.join(splits_path,'test_df.csv'))
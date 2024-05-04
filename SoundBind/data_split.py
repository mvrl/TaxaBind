import pandas as pd
import os
import numpy as np

data_path = "/storage1/fs1/jacobsn/Active/proj_smart/inat_image_sounds/"
meta_df = pd.read_csv(os.path.join(data_path,"metadata.csv"))

##Split IDs
train_df, validate_df, test_df = np.split(meta_df.sample(frac=1, random_state=42), [int(.85*len(meta_df)), int(.9*len(meta_df))])
print("sample count:train/val/test for ratio 85:5:10",(len(train_df),len(validate_df)),len(test_df)) #sample count:train/val/test for ratio 85:5:10 (75402, 4436) 8871
splits_path = data_path
train_df.to_csv(os.path.join(splits_path,'train_df.csv'))
validate_df.to_csv(os.path.join(splits_path,'validate_df.csv'))
test_df.to_csv(os.path.join(splits_path,'test_df.csv'))
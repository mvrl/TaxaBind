import os
import pandas as pd
from tqdm import tqdm
import torchaudio
from PIL import Image

data_path = "/storage1/jacobsn/Active/proj_smart/inat_image_sounds/"
meta_df = pd.read_csv(os.path.join(data_path,"metadata.csv"))
failed_ids = []
for i in tqdm(range(len(meta_df))):
    sample = meta_df.iloc[i]
    id = sample.id
    sound_format = sample.sound_format
    path_to_audio = os.path.join(data_path,"sounds",str(id)+"."+sound_format)
    image_path = os.path.join(data_path,"images",str(id)+".jpg")
    try:
        image = Image.open(image_path)
        track, sr = torchaudio.load(path_to_audio,format=sound_format) 
    except:
        failed_ids.append(id)
        print("Failed for:",len(failed_ids))

df = pd.DataFrame(columns=['id'])
df['id'] = failed_ids
df.to_csv("/storage1/jacobsn/Active/proj_smart/inat_image_sounds/failed_ids.csv")
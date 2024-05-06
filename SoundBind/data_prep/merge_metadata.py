##This script merges csv including samples for which both sound and image are available

import pandas as pd
import os
from tqdm import tqdm

data_path = "/scratch/k.subash/data/inat_image_sounds"
df = pd.read_csv(os.path.join(data_path,"observations-426986_clean.csv"))

images = os.listdir(os.path.join(data_path,"images"))
sounds = os.listdir(os.path.join(data_path,"sounds"))

images_ids = [int(i.split(".")[0]) for i in images]
sounds_ids = [int(i.split(".")[0]) for i in sounds]

common_ids = set(images_ids).intersection(sounds_ids)
df_final = df[df['id'].isin(common_ids)]

sound_formats = []
for i in tqdm(range(len(df_final))):
    row = df_final.iloc[i]
    sound_id = row.id
    url = row.sound_url
    #.m4a?1502824247#
    sound_format = url.split("/")[-1].split(".")[1].split("?")[0]
    sound_formats.append(sound_format)

df_final['sound_format'] = sound_formats
# {'m4a': 34003, 'wav': 30325, 'mp3': 24381, 'mpga': 596, '3gp': 52, 'mp4': 23, 'x-hx-aac-adts': 8, 'mpg': 1, '3gpp': 1, 'mpeg': 1}
df_final = df_final[df_final['sound_format'].isin(['m4a','wav','mp3'])]

df_final.to_csv(os.path.join(data_path,"metadata.csv"))
# exec(os.environ.get("DEBUG"))
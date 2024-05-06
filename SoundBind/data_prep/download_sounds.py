#This script downloads sounds from iNaturalist

import requests
import os
import pandas as pd
from tqdm import tqdm
import time


audio_path = "/scratch/k.subash/data/inat_image_sounds/sounds"
metadata_path = "/scratch/k.subash/data/inat_image_sounds/observations-426986.csv" #from inat download
final_meta_path = "/scratch/k.subash/data/inat_image_sounds/observations-426986_clean.csv"

# Using the following function, simple cleaning of metadata was already done.
def clean_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=['id', 'sound_url', 'image_url','latitude', 'longitude'])
    df.to_csv(final_meta_path)
    return df

def download_sound(url, filename):
    if not os.path.exists(filename):
        try:
            time.sleep(1)
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                return True
            else:
                print('Download failed for {}'.format(filename,url))
                return False
        except:
            print('Download failed for {}'.format(filename,url))
            return False
    else:
        return True


df = clean_metadata(metadata_path)

for i in tqdm(range(len(df)), desc='Downloading for iNat sounds'):
    row = df.iloc[i]
    sound_id = row.id
    url = row.sound_url
    #.m4a?1502824247#
    sound_format = url.split("/")[-1].split(".")[1].split("?")[0]
    filename = os.path.join(audio_path,str(sound_id)+'.'+sound_format)
    # exec(os.environ.get("DEBUG"))
    result = download_sound(url,filename)
    # import code;code.interact(local=dict(globals(), **locals()));
print('Download complete!')

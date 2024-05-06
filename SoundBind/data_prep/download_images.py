#This script downloads images from iNaturalist

import requests
import os
import pandas as pd
from tqdm import tqdm
import time

image_path = "/scratch/k.subash/data/inat_image_sounds/images" #create this path
metadata_path = "/scratch/k.subash/data/inat_image_sounds/observations-426986.csv" #from inat download
final_meta_path = "/scratch/k.subash/data/inat_image_sounds/observations-426986_clean.csv"

# Using the following function, simple cleaning of metadata was already done.
def clean_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=['id', 'sound_url', 'image_url','latitude', 'longitude'])
    df.to_csv(final_meta_path)

clean_metadata(metadata_path)

def download_image(url, filename):
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


# Read image IDs from the CSV file
image_ids = list(pd.read_csv(final_meta_path)['id'])
image_urls = list(pd.read_csv(final_meta_path)['image_url'])

meta_df = pd.read_csv(final_meta_path)[['id','image_url']]


for i in tqdm(range(len(meta_df)), desc='Downloading images'):
    row = meta_df.iloc[i]
    image_id = row.id
    url = row.image_url
    image_format = url.split("/")[-1].split(".")[1].lower()
    filename = os.path.join(image_path,str(image_id)+'.'+image_format)
    result = download_image(url,filename)
print('Download complete!')

# ##Usage: 
# ##python download_images.py
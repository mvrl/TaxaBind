from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image
from datetime import datetime
from torchvision.transforms import v2
import torch
import numpy as np

class INatDataset(Dataset):
    def __init__(self, work_dir, json_path, env_path):
        self.work_dir = work_dir
        self.json = json.load(open(os.path.join(self.work_dir, json_path)))
        self.images = self.json['images']
        self.annot = self.json['annotations']
        for i in range(len(self.images)):
            assert self.images[i]['id'] == self.annot[i]['id']
            self.images[i]['label'] = self.annot[i]['category_id']
        self.filtered_json = [d for d in self.images if d['latitude'] is not None and d['longitude'] is not None]
        self.env = torch.tensor(np.load(env_path))
        self.env[torch.isnan(self.env)] = 0.0
        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                # transforms.CenterCrop((224, 224)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.5),
                transforms.GaussianBlur(5, (0.01, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.filtered_json)
    def __getitem__(self, idx):
        try:
            latitude = float(self.filtered_json[idx]['latitude'])
            longitude = float(self.filtered_json[idx]['longitude'])
            neg_lat = torch.rand(2)
            neg_lat[0] = neg_lat[0]*180.0 - 90.0
            neg_lat[1] = neg_lat[1]*360.0 - 180.0
            x_cell = int((longitude+180.0)*self.env.shape[1]/360.0)
            y_cell = int((-latitude+90.0)*self.env.shape[0]/180.0)
            env_feats = self.env[y_cell, x_cell, :]
            x_cell_neg = int((neg_lat[1]+180.0)*self.env.shape[1]/360.0)
            y_cell_neg = int((-neg_lat[0]+90.0)*self.env.shape[0]/180.0)
            env_feats_neg = self.env[y_cell_neg, x_cell_neg, :]

        except Exception as e:
            print(e)
            #print(self.json['images'][idx])
            while True:
                try:
                    idx = torch.randint(0, len(self.filtered_json), (1,)).item()
                    latitude = float(self.filtered_json[idx]['latitude'])
                    longitude = float(self.filtered_json[idx]['longitude'])
                    neg_lat = torch.rand(2)
                    neg_lat[0] = neg_lat[0]*180.0 - 90.0
                    neg_lat[1] = neg_lat[1]*360.0 - 180.0
                    x_cell = int((longitude+180.0)*self.env.shape[1]/360.0)
                    y_cell = int((-latitude+90.0)*self.env.shape[0]/180.0)
                    env_feats = self.env[y_cell, x_cell, :]
                    x_cell_neg = int((neg_lat[1]+180.0)*self.env.shape[1]/360.0)
                    y_cell_neg = int((-neg_lat[0]+90.0)*self.env.shape[0]/180.0)
                    env_feats_neg = self.env[y_cell_neg, x_cell_neg, :]
                    break
                except:
                    #print(self.json['images'][idx])
                    continue
        img_path = os.path.join(self.work_dir, self.filtered_json[idx]['file_name'])
        img = self.transform(Image.open(img_path))
        return img, env_feats, env_feats_neg, self.filtered_json[idx]['label']


if __name__=='__main__':
    inat_data = INatDataset('../metaformer', 'train.json')
    print(len(inat_data))
    import code; code.interact(local=locals())
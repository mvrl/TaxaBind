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
    def __init__(self, work_dir, json_path, env_path, mode='train'):
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
        self.species_text = list(set([" ".join(d['file_name'].split("/")[1].split("_")[1:]) for d in self.filtered_json]))
        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.GaussianBlur(5, (0.01, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.filtered_json)
    def __getitem__(self, idx):
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
        img_path = os.path.join(self.work_dir, self.filtered_json[idx]['file_name'])
        img = self.transform(Image.open(img_path))
        species_text = " ".join(self.filtered_json[idx]['file_name'].split("/")[1].split("_")[1:])
        return img, env_feats, env_feats_neg, self.species_text.index(species_text)
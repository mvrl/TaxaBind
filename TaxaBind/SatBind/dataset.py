from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image
from datetime import datetime
from torchvision.transforms import v2
import torch
import numpy as np
import glob

class SatNatDataset(Dataset):
    def __init__(self, img_dir, imo_dir, json_path):
        self.img_dir = img_dir
        self.imo_dir = imo_dir

        self.json = json.load(open(json_path, 'r'))
        self.images = self.json['images']
        self.annot = self.json['annotations']
        for i in range(len(self.images)):
            assert self.images[i]['id'] == self.annot[i]['id']
            self.images[i]['label'] = self.annot[i]['category_id']
        self.filtered_json = [d for d in self.images if d['latitude'] is not None and d['longitude'] is not None]
        #self.filtered_json = [d for d in self.filtered_json if os.path.exists(os.path.join(self.imo_dir, str(d['id'])+'_'+str(d['latitude'])+'_'+str(d['longitude'])+'.jpeg'))]
        self.species_text = list(set([" ".join(d['file_name'].split("/")[1].split("_")[1:]) for d in self.filtered_json]))
        self.sat_paths = {d['id']: str(d['id'])+'_'+str(d['latitude'])+'_'+str(d['longitude'])+'.jpeg' for d in self.filtered_json}
        self.img_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                #transforms.CenterCrop((224, 224)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.5),
                transforms.GaussianBlur(5, (0.01, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.imo_transform = transforms.Compose([
            transforms.Resize((256,256)),
            #transforms.CenterCrop((224, 224)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomVerticalFlip(0.5),
            transforms.GaussianBlur(5, (0.01, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.filtered_json)
    def __getitem__(self, idx):
        flag = 1
        try:
            img_path = os.path.join(self.img_dir, self.filtered_json[idx]['file_name'])
            #img_path = img_path.replace('train_mini', 'train')
            imo_path = os.path.join(self.imo_dir, self.sat_paths[self.filtered_json[idx]['id']])
            img = self.img_transform(Image.open(img_path))
            imo = self.imo_transform(Image.open(imo_path))
        except Exception as e:
            flag = 0
            #print(e)
            while True:
               #print(e)
                try:
                    idx = torch.randint(0, len(self.filtered_json), (1,)).item()
                    #idx += 1
                    img_path = os.path.join(self.img_dir, self.filtered_json[idx]['file_name'])
                    #img_path = img_path.replace('train_mini', 'train')
                    imo_path = os.path.join(self.imo_dir, self.sat_paths[self.filtered_json[idx]['id']])
                    img = self.img_transform(Image.open(img_path))
                    imo = self.imo_transform(Image.open(imo_path))
                    break
                except:
                    continue
        species_text = " ".join(self.filtered_json[idx]['file_name'].split("/")[1].split("_")[1:])
        return img, imo, self.filtered_json[idx]['label'], species_text, self.species_text.index(species_text), flag

class OV(Dataset):
    def __init__(self, imo_dir, json_path):
        self.imo_dir = imo_dir

        self.json = json.load(open(json_path, 'r'))
        self.images = self.json['images']
        self.annot = self.json['annotations']
        for i in range(len(self.images)):
            assert self.images[i]['id'] == self.annot[i]['id']
            self.images[i]['label'] = self.annot[i]['category_id']
        self.filtered_json = [d for d in self.images if d['latitude'] is not None and d['longitude'] is not None]
        self.filtered_json = [d for d in self.filtered_json if os.path.exists(os.path.join(self.imo_dir, str(d['id'])+'_'+str(d['latitude'])+'_'+str(d['longitude'])+'.jpeg'))]
        self.species_text = list(set([" ".join(d['file_name'].split("/")[1].split("_")[1:]) for d in self.filtered_json]))
        self.sat_paths = {d['id']: str(d['id'])+'_'+str(d['latitude'])+'_'+str(d['longitude'])+'.jpeg' for d in self.filtered_json}
        self.imo_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224, 224)),
            #transforms.RandomCrop((224, 224)),
            #transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomVerticalFlip(0.5),
            #transforms.GaussianBlur(5, (0.01, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.plot_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.filtered_json)
    def __getitem__(self, idx):
        imo_path = os.path.join(self.imo_dir, self.sat_paths[self.filtered_json[idx]['id']])
        img = Image.open(imo_path)
        ov_img = self.imo_transform(img)
        plot_img = self.plot_transform(img)
        species_text = " ".join(self.filtered_json[idx]['file_name'].split("/")[1].split("_")[1:])
        return ov_img, plot_img, self.filtered_json[idx]['label'], species_text, self.species_text.index(species_text)


class GL(Dataset):
    def __init__(self, img_dir, imo_dir, json_path):
        self.img_dir = img_dir
        self.imo_dir = imo_dir
        self.json = json.load(open(json_path, 'r'))
        self.images = self.json['images']
        self.annot = self.json['annotations']
        for i in range(len(self.images)):
            assert self.images[i]['id'] == self.annot[i]['id']
            self.images[i]['label'] = self.annot[i]['category_id']
        self.filtered_json = [d for d in self.images if d['latitude'] is not None and d['longitude'] is not None]
        self.filtered_json = [d for d in self.filtered_json if os.path.exists(os.path.join(self.imo_dir, str(d['id'])+'_'+str(d['latitude'])+'_'+str(d['longitude'])+'.jpeg'))]
        self.species_text = list(set([" ".join(d['file_name'].split("/")[1].split("_")[1:]) for d in self.filtered_json]))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.plot_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.filtered_json)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filtered_json[idx]['file_name'])
        img = Image.open(img_path)
        gl_img = self.img_transform(img)
        plot_img = self.plot_transform(img)
        species_text = " ".join(self.filtered_json[idx]['file_name'].split("/")[1].split("_")[1:])
        return gl_img, plot_img, self.filtered_json[idx]['label'], species_text, self.species_text.index(species_text)


if __name__ == '__main__':
    img_dir = '/scratch/s.sastry/ecobind_data/'
    imo_dir = '/scratch/s.sastry/ecobind_satellite/ecobind_sentinel/images/sentinel/'
    json_path = '/scratch/s.sastry/ecobind_data/train_mini.json'
    import code; code.interact(local=dict(globals(), **locals()))
    ds = SatNatDataset(img_dir, imo_dir, json_path)
    img, imo, label = ds[0]
    
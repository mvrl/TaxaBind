from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image
from datetime import datetime
from torchvision.transforms import v2
import torch


class INatDataset(Dataset):
    def __init__(self, work_dir, json_path):
        self.work_dir = work_dir
        self.json = json.load(open(os.path.join(self.work_dir, json_path)))
        self.images = self.json['images']
        self.annot = self.json['annotations']
        for i in range(len(self.images)):
            assert self.images[i]['id'] == self.annot[i]['id']
            self.images[i]['label'] = self.annot[i]['category_id']
        self.filtered_json = [d for d in self.images if d['latitude'] is not None and d['longitude'] is not None]
        self.species_text = list(set([" ".join(d['file_name'].split("/")[1].split("_")[1:]) for d in self.filtered_json]))
        self.transform = transforms.Compose([
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
    def __len__(self):
        return len(self.filtered_json)
    def __getitem__(self, idx):
        flag = 1
        try:
            latitude = float(self.filtered_json[idx]['latitude'])
            longitude = float(self.filtered_json[idx]['longitude'])
            date = self.filtered_json[idx]['date'].split(" ")[0]
            year = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%Y'))
            month = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%m'))
            day = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%d'))
        except Exception as e:
            #print(e)
            #print(self.json['images'][idx])
            flag = 0
            while True:
                try:
                    idx = torch.randint(0, len(self.filtered_json), (1,)).item()
                    latitude = float(self.filtered_json[idx]['latitude'])
                    longitude = float(self.filtered_json[idx]['longitude'])
                    date = self.filtered_json[idx]['date'].split(" ")[0]
                    year = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%Y'))
                    month = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%m'))
                    day = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%d'))
                    break
                except:
                    #print(self.json['images'][idx])
                    continue
        img_path = os.path.join(self.work_dir, self.filtered_json[idx]['file_name'])
        img = self.transform(Image.open(img_path))
        species_text = " ".join(self.filtered_json[idx]['file_name'].split("/")[1].split("_")[1:])
        return img, torch.Tensor([latitude, longitude]), self.filtered_json[idx]['label'], species_text, self.species_text.index(species_text), year, month, day, flag


if __name__=='__main__':
    inat_data = INatDataset('../ecobind_data', 'train.json')
    print(len(inat_data))
    import code; code.interact(local=locals())
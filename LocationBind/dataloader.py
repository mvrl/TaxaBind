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
        self.json = json.load(open(os.path.join(self.work_dir, json_path)))['images']
        self.filtered_json = [d for d in self.json if d['latitude'] is not None and d['longitude'] is not None]
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
            date = self.filtered_json[idx]['date'].split(" ")[0]
            year = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%Y'))
            month = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%m'))
            day = int(datetime.strptime(date, '%Y-%m-%d').date().strftime('%d'))
        except Exception as e:
            print(e)
            #print(self.json['images'][idx])
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
        return img, torch.Tensor([latitude, longitude]), year, month, day


if __name__=='__main__':
    inat_data = INatDataset('../metaformer', 'train.json')
    print(len(inat_data))
    import code; code.interact(local=locals())
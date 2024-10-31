from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from config import cfg
from sound_encoder import get_audio_clap
import torch
from tqdm import tqdm

#img, sound, year, month, day
class INatDataset(Dataset):
    def __init__(self,
                 data_file,
                 mode='train'): 
        self.data_file = pd.read_csv(data_file)
        if mode=='train':
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
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        self.species_text = self.data_file['scientific_name'].tolist()
        self.species_classes = list(set(self.species_text))

    def __len__(self):
        return len(self.data_file)
        
    def get_sample(self,idx):
        sample = self.data_file.iloc[idx]
        id = sample.id
        sound_format = sample.sound_format
        image_path = os.path.join(cfg['data_path'],"images",str(id)+".jpg")
        sound_path = os.path.join(cfg['data_path'],"sounds_mp3",str(id)+"."+'mp3')
        sound = get_audio_clap(sound_path, sound_format)
        
        for k in sound.keys():
            sound[k] = sound[k].squeeze(0)
        image = self.transform(Image.open(image_path))

        return image, sound

    def __getitem__(self, idx):
        image, sound = self.get_sample(idx)
        return image, sound, self.species_classes.index(self.data_file.iloc[idx]['scientific_name'])

from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from config import cfg
from sound_encoder import get_audio_clap
import torch

#img, sound, year, month, day
class INatDataset(Dataset):
    def __init__(self,
                 data_file): 
        self.data_file = pd.read_csv(data_file)
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
        return len(self.data_file)
        
    def get_sample(self,idx):
        sample = self.data_file.iloc[idx]
        id = sample.id
        sound_format = sample.sound_format
        image_path = os.path.join(cfg['data_path'],"images",str(id)+".jpg")
        sound_path = os.path.join(cfg['data_path'],"sounds",str(id)+"."+sound_format)
        sound = get_audio_clap(sound_path)
        
        for k in sound.keys():
            sound[k] = sound[k].squeeze(0)
        image = self.transform(Image.open(image_path))

        return image, sound

    def __getitem__(self, idx):
        try:
            image, sound = self.get_sample(idx)
        except:
            while True:
                try:
                    idx = torch.randint(0, len(self.data_file), (1,)).item()
                    image, sound = self.get_sample(idx)
                    break
                except:
                    continue

        return image, sound


if __name__=='__main__':
    inat_data = INatDataset(data_file=cfg['train_df'])
    print(len(inat_data))
    train_loader = torch.utils.data.DataLoader(inat_data,
                                            num_workers=0, batch_size=8, shuffle=True, drop_last=False,pin_memory=True)
    for i in range(10):
        images, sounds = next(iter(train_loader))
        print(images.shape, sounds['input_features'].shape, sounds['is_longer'].shape)
        # import code; code.interact(local=locals())
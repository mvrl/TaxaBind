from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from config import cfg
from sound_encoder import get_audio_clap
import torch
from tqdm import tqdm
from get_full_name import get_taxonomic_name

#img, sound, year, month, day
class INatDataset(Dataset):
    def __init__(self,
                 data_file): 
        self.data_file = pd.read_csv(data_file)
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
        self.species_text = self.data_file['scientific_name'].tolist()
        self.species_classes = list(set(self.species_text))
        self.species_text = [get_taxonomic_name(name) for name in self.species_classes]
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
        flag = 1
        try:
            image, sound = self.get_sample(idx)
        except Exception as e:
            flag = 0
            #print(e)
            while True:
                try:
                    idx = torch.randint(0, len(self.data_file), (1,)).item()
                    image, sound = self.get_sample(idx)
                    break
                except:
                    continue

        return image, sound, flag, self.species_classes.index(self.data_file.iloc[idx]['scientific_name'])


if __name__=='__main__':
    inat_data = INatDataset(data_file=cfg['train_df'])
    print(len(inat_data))
    train_loader = torch.utils.data.DataLoader(inat_data,
                                            num_workers=8, batch_size=8, shuffle=False, drop_last=False, pin_memory=True)
    # for i in range(10):
    #     images, sounds = next(iter(train_loader))
    #     print(images.shape, sounds['input_features'].shape, sounds['is_longer'].shape)
    #     # import code; code.interact(local=locals())
    num_points = 0
    for i, batch in tqdm(enumerate(train_loader)):
        num_points += batch[-1].sum()
        if i%100==0:
            print(num_points)
    print(num_points)

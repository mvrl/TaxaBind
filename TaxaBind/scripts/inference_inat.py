from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import os
from PIL import Image
from datetime import datetime
from torchvision.transforms import v2
import torch
import numpy as np
import glob
from tqdm import tqdm

class iNatDataset(Dataset):
    def __init__(self, img_dir, json_path):
        self.img_dir = img_dir

        self.json = json.load(open(json_path, 'r'))
        self.images = self.json['images']
        self.annot = self.json['annotations']
        for i in range(len(self.images)):
            assert self.images[i]['id'] == self.annot[i]['id']
            self.images[i]['label'] = self.annot[i]['category_id']
        self.filtered_json = [d for d in self.images if d['latitude'] is not None and d['longitude'] is not None]
        self.species_text = list(set([" ".join(d['file_name'].split("/")[1].split("_")[1:]) for d in self.filtered_json]))
        self.img_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.filtered_json)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filtered_json[idx]['file_name'])
        img = self.img_transform(Image.open(img_path))
        species_text = " ".join(self.filtered_json[idx]['file_name'].split("/")[1].split("_")[1:])
        return img, self.filtered_json[idx]['label'], species_text, self.species_text.index(species_text)


def test_inat(bioclip, tokenizer, img_dir, json_path, batch_size=512, num_workers=8, device='cuda'):
    inat_data = iNatDataset(img_dir, json_path)
    inat_loader = DataLoader(inat_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    text_embeds = []
    species_text = inat_data.species_text

    for i in tqdm.tqdm(range(65)):
        _, feats_1, _ = model(text=tokenizer(species_text[len(species_text)//64*i:len(species_text)//64*(i+1)]).to(device))
        text_embeds.append(feats_1.detach().cpu())
        del feats_1
    text_embeds = torch.cat(text_embeds)

    res_bioclip = 0

    for i, batch in tqdm(enumerate(inat_loader)):
        img, label, species_text, species_idx = batch
        img = img.to(device)
        label = label.to(device)
        species_text = species_text.to(device)
        species_idx = species_idx.to(device)

        img_embeds, *_ = bioclip(img)
        img_embeds = img_embeds.detach().cpu()

        logits = torch.matmul(img_embeds, text_embeds.t())
        res_bioclip += torch.sum(torch.argmax(logits, dim=1) == species_idx)
    
    return res_bioclip / len(inat_data)
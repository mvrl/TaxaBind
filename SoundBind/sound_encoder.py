#Hugging face way of loading AudioCLAP model
from transformers import ClapProcessor
from transformers import ClapAudioModelWithProjection
import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np
import torchaudio
import os

processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
SAMPLE_RATE = 48000

def get_audio_clap(path_to_audio,padding="repeatpad",truncation="fusion"):
    track, sr = torchaudio.load(path_to_audio)
    track = track.mean(axis=0)
    track = torchaudio.functional.resample(track, orig_freq=sr, new_freq=SAMPLE_RATE)
    output = processor(audios=track, sampling_rate=SAMPLE_RATE, max_length_s=10, return_tensors="pt",padding=padding,truncation=truncation)
    return output


class CLAP_audiomodel_withProjection(pl.LightningModule):
    def __init__(self,freeze=False):
        super().__init__()
        if freeze:
            self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").eval()
            for params in self.model.parameters():
                params.requires_grad=False
        else:
            self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").train()
    def forward(self,audio):
        batch_embeddings_audio = self.model(**audio)['audio_embeds']
        return batch_embeddings_audio
    
if __name__ == '__main__':
    path_to_audio ="/storage1/fs1/jacobsn/Active/proj_smart/inat_image_sounds/sounds_mp3/100002768.mp3"
    sample =  get_audio_clap(path_to_audio)
    print(sample.keys())
    
    sample['input_features'] = torch.concat([sample['input_features'],sample['input_features']],axis=0)
    sample['is_longer'] = torch.concat([sample['is_longer'],sample['is_longer']],axis=0)
    print(sample['input_features'].shape,sample['is_longer'].shape) #torch.Size([2, 4, 1001, 64]), torch.Size([2, 1])
    model = CLAP_audiomodel_withProjection(freeze=False)
    audio_feat = model(sample)
    print(audio_feat.shape) #torch.Size([2, 512])
    

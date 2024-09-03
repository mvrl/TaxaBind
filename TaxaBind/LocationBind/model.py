import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
from location_encoder import LocationEncoder
import numpy as np
from torch.utils.data import DataLoader

def create_pairwise_mask(labels):
    labels = labels.reshape(-1)
    num_samples = len(labels)
    pairwise_mask = torch.zeros(num_samples, num_samples).to(labels.device)

    for i in range(num_samples):
        pairwise_mask[i, :] = (labels == labels[i])

    return pairwise_mask

def clip_loss(similarity: torch.Tensor, label) -> torch.Tensor:
    label_mask = 1 - create_pairwise_mask(label) + torch.eye(similarity.shape[0]).to(similarity.device)
    label_mask = torch.cat((label_mask, torch.ones(label_mask.shape).to(label_mask.device)), dim=-1)
    similarity[label_mask==0] = -float('inf')
    overhead_img_loss = contrastive_loss(similarity, compute_neg=True)
    ground_img_loss = contrastive_loss(similarity.t())
    return 0.4*torch.mean(torch.sum(overhead_img_loss, dim=-1)) + 0.6*torch.mean(torch.sum(ground_img_loss, dim=-1))

def contrastive_loss(logits: torch.Tensor, compute_neg=False) -> torch.Tensor:
    if compute_neg:
        gt = torch.cat((torch.eye(logits.shape[0]), torch.zeros(logits.shape[0], logits.shape[0])), dim=-1).to(logits.device)
        return - gt*torch.log(logits.softmax(-1)+1e-6)
    else:
        gt = torch.eye(logits.shape[1], device=logits.device)
        return - gt*torch.log(logits[:logits.shape[1]].softmax(-1)+1e-6)

class LocationBind(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model, *_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.location_encoder = LocationEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size = kwargs.get('batch_size', 512)
        self.lr = kwargs.get('lr', 1e-4)
        self.model.train()

    def forward(self, image, location, sample_neg=False):
        image_embeds, *_ = self.model(image)
        if sample_neg:
            neg_lat = torch.rand(image_embeds.shape[0], 2).to(location.device)
            neg_lat[:, 0] = neg_lat[:, 0]*180.0 - 90.0
            neg_lat[:, 1] = neg_lat[:, 1]*360.0 - 180.0
            location = torch.cat((location, neg_lat))
        location_embeds = torch.nn.functional.normalize(self.location_encoder(location), dim=-1)
        return image_embeds, location_embeds
    
    def shared_step(self, batch):
        image, location, label, *_ = batch
        image_embeds, location_embeds = self(image, location, sample_neg=True)
        logit_scale = self.logit_scale.exp()
        logits_per_img = torch.matmul(image_embeds,location_embeds.t())*logit_scale
        cross_contrastive_loss = clip_loss(logits_per_img, label)
        return cross_contrastive_loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('temperature', self.logit_scale.data, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def on_train_batch_end(self,outputs,batch, batch_idx):
        if self.logit_scale.data > np.log(100):
            self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, np.log(100))

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=16,
                          shuffle=True,
                          persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=16,
                          shuffle=True,
                          persistent_workers=False)
    
    def configure_optimizers(self):
        params = self.parameters()
        self.optim = torch.optim.AdamW(params,
                                       lr=self.lr,
                                       betas=(0.9,0.98),
                                       eps=1e-6
                                    )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optim,
            T_0=20
        )
        return [self.optim], [self.scheduler]
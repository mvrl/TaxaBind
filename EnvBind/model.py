import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    overhead_img_loss = contrastive_loss(similarity, compute_neg=True)
    ground_img_loss = contrastive_loss(similarity.t())
    return 0.4*overhead_img_loss + 0.6*ground_img_loss

def contrastive_loss(logits: torch.Tensor, compute_neg=False) -> torch.Tensor:
    if compute_neg:
        return nn.functional.cross_entropy(logits, torch.cat((torch.eye(logits.shape[0]), torch.zeros(logits.shape[0], logits.shape[0])), dim=-1).to(logits.device))
    else:
        return nn.functional.cross_entropy(logits[:logits.shape[1]], torch.arange(logits.shape[1], device=logits.device))

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.1)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class ResidualFCNet(nn.Module):

    def __init__(self, num_inputs, num_filts, depth=4):
        super(ResidualFCNet, self).__init__()
        layers = []
        layers.append(nn.Linear(num_inputs, num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = torch.nn.Sequential(*layers)

    def forward(self, x):
        env_emb = self.feats(x)
        return env_emb

class EnvBind(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model, *_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        for param in self.model.parameters():
            param.requires_grad = False
        self.env_encoder = ResidualFCNet(num_inputs=20, num_filts=512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size = kwargs.get('batch_size', 1024)
        self.lr = kwargs.get('lr', 1e-4)

    def forward(self, image, env_feats, env_feats_neg):
        with torch.no_grad():
            image_embeds, *_ = self.model(image)
        env_feats = torch.cat((env_feats, env_feats_neg))
        env_embeds = torch.nn.functional.normalize(self.env_encoder(env_feats.float()), dim=-1)
        return image_embeds, env_embeds
    
    def shared_step(self, batch):
        image, env_feats, env_feats_neg = batch
        image_embeds, env_embeds = self(image, env_feats, env_feats_neg)
        logit_scale = self.logit_scale.exp()
        logits_per_img = torch.matmul(image_embeds,env_embeds.t())*logit_scale
        cross_contrastive_loss = clip_loss(logits_per_img)
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
import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    overhead_img_loss = contrastive_loss(similarity)
    ground_img_loss = contrastive_loss(similarity.t())
    return (overhead_img_loss + ground_img_loss) / 2.0

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits[:logits.shape[1]], torch.arange(logits.shape[1], device=logits.device))

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

    def forward(self, image, env_feats):
        with torch.no_grad():
            image_embeds, *_ = self.model(image)
        env_embeds = torch.nn.functional.normalize(self.env_encoder(env_feats), dim=-1)
        return image_embeds, env_embeds
    
    def shared_step(self, batch):
        image, env_feats, *_ = batch
        image_embeds, env_embeds = self(image, env_feats)
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
                          shuffle=False,
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
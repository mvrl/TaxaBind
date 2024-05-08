import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection
from dataset import SatNatDataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def create_pairwise_mask(labels):
    labels = labels.reshape(-1)
    num_samples = len(labels)
    pairwise_mask = torch.zeros(num_samples, num_samples).to(labels.device)

    for i in range(num_samples):
        pairwise_mask[i, :] = (labels == labels[i])

    return pairwise_mask

def clip_loss(similarity: torch.Tensor, label) -> torch.Tensor:
    label_mask = 1 - create_pairwise_mask(label) + torch.eye(similarity.shape[0]).to(similarity.device)
    similarity[label_mask==0] = -float('inf')
    overhead_img_loss = contrastive_loss(similarity)
    ground_img_loss = contrastive_loss(similarity.t())
    return 0.5*torch.mean(torch.sum(overhead_img_loss, dim=-1)) + 0.5*torch.mean(torch.sum(ground_img_loss, dim=-1))

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    gt = torch.eye(logits.shape[0], device=logits.device)
    return - gt*torch.log(logits.softmax(-1)+1e-6)

class SatBind(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        #initialize bio CLIP with frozen weights
        self.bio_model, *_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        for param in self.bio_model.parameters():
            param.requires_grad = False
        
        #initialize CLIP with trainable weights
        self.imo_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch16').train()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size = kwargs.get('batch_size', 10)
        self.lr = kwargs.get('lr', 1e-4)
    

    def forward(self, batch):
        img, imo, label = batch
        #compute bioclip embeddings
        img_embeds, *_ = self.bio_model(img)
        
        #compute overhead embeddings
        imo_embeds = self.imo_encoder(imo).image_embeds

        return img_embeds, imo_embeds, label

    
    def shared_step(self, batch):
        
        img_embeds, imo_embeds, label = self(batch)
        #normalize embeddings
        #img embeds is already normalized
        img_embeds = img_embeds
        imo_embeds = torch.nn.functional.normalize(imo_embeds, dim=-1)
        
        #exponentiate the log of temperrature
        logit_scale = self.logit_scale.exp()

        #compute similarity 
        img_to_imo_sim = img_embeds @ imo_embeds.t() * logit_scale
        
        loss = clip_loss(img_to_imo_sim, label) 
        return loss     


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.log('temperature', self.logit_scale.data, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
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

if __name__ == '__main__':
    img_dir = '/scratch/s.sastry/ecobind_data/'
    imo_dir = '/scratch/s.sastry/ecobind_satellite/ecobind_sentinel/images/sentinel/'
    imo_dir_val = '/scratch/s.sastry/ecobind_satellite/ecobind_val_sentinel/images/sentinel/'
    train_json_path = '/scratch/s.sastry/ecobind_data/train_mini.json'
    val_json_path = '/scratch/s.sastry/ecobind_data/val.json'
    
    #define dataset
    train_dataset = SatNatDataset(img_dir, imo_dir, train_json_path)
    val_dataset = SatNatDataset(img_dir, imo_dir_val, val_json_path)

    #define model
    model = SatBind(train_dataset=train_dataset, val_dataset=val_dataset)
    torch.cuda.empty_cache()
    logger = WandbLogger(project="Sat-Bind", name="demo_run", mode='disabled')
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='satbind-{epoch:02d}-{val_loss:.2f}',
        mode='min'
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices='0,', 
        max_epochs=1500,
        num_nodes=1,
        callbacks=[checkpoint],
        accumulate_grad_batches=8,
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=0.25,
        fast_dev_run=1
        )
    trainer.fit(model)
    
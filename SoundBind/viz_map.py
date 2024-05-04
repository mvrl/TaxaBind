from model import LocationBind
import matplotlib.pyplot as plt
import torch
from sklearn import decomposition
from skimage import exposure
import numpy as np
from SoundBind.sound_encoder import LocationEncoder

model = LocationBind.load_from_checkpoint("./checkpoints/locbind-epoch=00-val_loss=5.59.ckpt", train_dataset=None, val_dataset=None)
model = model.location_encoder.eval().cuda()
# model = LocationEncoder().cuda()

x, y = torch.linspace(-180.0, 180.0, 500), torch.linspace(90.0, -90.0, 250)
lat, lon = torch.meshgrid(y, x)
coords = torch.stack((lat, lon), dim=-1).reshape(-1, 2)

feats = model(coords.float().cuda()).detach().cpu().numpy()
f_mu = feats.mean(0)
f_std = feats.std(0)
feats = feats - f_mu
feats = feats / f_std

dsf = decomposition.FastICA(n_components=9, random_state=42, whiten='unit-variance', max_iter=1000)
dsf.fit(feats)

feats_ds = dsf.transform(feats)

for cc in range(3):
    feats_ds[:, cc+6] = exposure.equalize_hist(feats_ds[:, cc+6])

op_im = feats_ds[:, 6:].reshape((250, 500, 3))
plt.imsave("ica.png", (op_im*255).astype(np.uint8))


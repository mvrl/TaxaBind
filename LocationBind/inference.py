from model import LocationBind
import matplotlib.pyplot as plt
import torch
from sklearn import decomposition
from skimage import exposure
import numpy as np
from location_encoder import LocationEncoder
import open_clip
from PIL import Image
from torchvision import transforms


model, *_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
model.eval()
location_encoder = LocationBind.load_from_checkpoint("./checkpoints/locbind-epoch=01-val_loss=5.58.ckpt", train_dataset=None, val_dataset=None)
location_encoder = location_encoder.location_encoder.eval().cuda()
x, y = torch.linspace(-180.0, 180.0, 500), torch.linspace(90.0, -90.0, 250)
lat, lon = torch.meshgrid(y, x)
coords = torch.stack((lat, lon), dim=-1).reshape(-1, 2)

img_path = '1200.jpeg'
img = Image.open(img_path)

transform = transforms.Compose([
                transforms.Resize((256, 256)),
                # transforms.CenterCrop((224, 224)),
                transforms.CenterCrop((224, 224)),
                #transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.5),
                #transforms.GaussianBlur(5, (0.01, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

img_tensor = transform(img).unsqueeze(0).cuda()
model = model.cuda()
with torch.no_grad():
    image_embeds, *_ = model(img_tensor)
    image_embeds = image_embeds.cpu().numpy()
    feats = torch.nn.functional.normalize(location_encoder(coords.float().cuda()), dim=-1).detach().cpu().numpy()

sims = feats @ image_embeds.T
sims = np.clip(sims, np.quantile(sims, 0.90), 1)
# plt.hist(sims, 20)
# plt.savefig("hist.png")
# plt.figure()
op_im = sims.reshape((250, 500))
#print(np.quantile(sims, 0.90))
plt.imsave("sims.png", op_im, cmap='Greens', vmax=sims.max(), vmin=0.0)

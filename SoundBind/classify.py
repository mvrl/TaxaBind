from model import LocationBind
import matplotlib.pyplot as plt
import torch
from sklearn import decomposition
from skimage import exposure
import numpy as np
from SoundBind.sound_encoder import LocationEncoder
import open_clip
from PIL import Image
from torchvision import transforms


model, *_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
model.eval()
location_encoder = LocationBind.load_from_checkpoint("./checkpoints/locbind-epoch=00-val_loss=5.54.ckpt", train_dataset=None, val_dataset=None)
location_encoder = location_encoder.location_encoder.eval().cuda()
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
text = tokenizer(["Animalia Chordata Aves Accipitriformes Accipitridae Haliaeetus leucocephalus", "Animalia Chordata Aves Sphenisciformes Spheniscidae Aptenodytes forsteri"]).cuda()

img_path = '1200 (1).jpeg'
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
    image_embeds, text_embeds, _ = model(img_tensor, text)
    print(torch.matmul(image_embeds, text_embeds.t()))
    # image_embeds = image_embeds.cpu().numpy()
    # import code; code.interact(local=locals())
    feats = torch.nn.functional.normalize(image_embeds + torch.nn.functional.normalize(location_encoder(torch.tensor([[23.966352044746348, 13.347993781010663]]).float().cuda()), dim=-1), dim=-1)
    print(torch.matmul(feats, text_embeds.t()))

# sims = feats @ image_embeds.T
# sims = np.clip(sims, np.quantile(sims, 0.95), 1)
# # plt.hist(sims, 20)
# # plt.savefig("hist.png")
# # plt.figure()
# op_im = sims.reshape((250, 500))
# #print(np.quantile(sims, 0.90))
# plt.imsave("sims.png", op_im, cmap='Greens', vmax=sims.max(), vmin=0.0)

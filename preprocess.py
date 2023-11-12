from torchvision.utils import save_image

from tqdm import tqdm

import os
import os.path as path
from glob import glob
import torchvision.transforms as T

from PIL import Image


mypath = "./img"
onlyfiles = glob("./img/*.jpeg")

preprocess = T.Compose([
   T.Resize(256),
   T.CenterCrop(224),
   T.ToTensor(),
#    T.Normalize(
#        mean=[0.485, 0.456, 0.406],
#        std=[0.229, 0.224, 0.225]
#    )
])

if not os.path.isdir('pimg'):
    os.mkdir('pimg')

for file in tqdm(onlyfiles, ncols=80): 
    img = Image.open(file)
    x = preprocess(img)
    save_image(x, file.replace('./img', './pimg'))
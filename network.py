import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split

from glob import glob
from tqdm import tqdm
from PIL import Image

from os import read

import src.utils as utils

device = 'mps'

birdfiles = sorted(glob("./pimg/*.jpeg"))
birds_in = []
transform = T.Compose([T.ToTensor()])

print("Creating bird tensors")
for file in tqdm(birdfiles, ncols=80):
    img = Image.open(file)
    birds_in.append(transform(img).to(device))
print()
            
birds_out, bird_order = utils.get_birds_out(device)
print()
    
print("Bird names: " + str(bird_order))
print()

class BirdDataset(Dataset):
    def __init__(self, birds_in, birds_out):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.birds_in = birds_in
        self.birds_out = birds_out

    def __len__(self):
        return len(self.birds_in)

    def __getitem__(self, idx):
        return [ self.birds_in[idx], self.birds_out[idx] ]
    

EPOCHS = 100
VALID_SPLIT = 0.3
BATCH_SIZE = 5

# no cherry or basil
birds_in_filtered = [ ]
birds_out_filtered = [ ]
for i in range(len(birds_in)):
    bird = utils.get_bird(birds_out[i], bird_order)
    if 'Cherry' not in bird and 'Basil' not in bird and 'Basil' != bird and 'Cherry' != bird:
        birds_in_filtered.append(birds_in[i])
        birds_out_filtered.append(birds_out[i])

DATASET = BirdDataset(birds_in, birds_out)
DATASET_FILTERED = BirdDataset(birds_in_filtered, birds_out_filtered)

model = utils.get_model(device, bird_order)

# Get index of first file path that doesn't exist
model_path_idx = utils.get_ckpt_index()
# If this is 0 there are no models to load
if model_path_idx > 0:
    model_path = utils.get_ckpt_path(model_path_idx - 1)
    
    # Ask to load
    if input("Load model from " + model_path + "? (y/n)") == "y":
        print("Yes, loading model.")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No, will not be loaded.")

optim = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
for i in range(EPOCHS):
    if i % 2 == 0:
        data = random_split(DATASET, [1-VALID_SPLIT, VALID_SPLIT])
    else:
        data = random_split(DATASET_FILTERED, [1-VALID_SPLIT, VALID_SPLIT])
    
    input_data = DataLoader(data[0], BATCH_SIZE, True)
    valid_data = DataLoader(data[1], BATCH_SIZE, True)
    
    loadingbar = tqdm(input_data, "Epoch " + str(i), ncols=80)
    avg_loss = 0
    model.train()
    for i, datapoint in enumerate(loadingbar):
        X_HAT = datapoint[0]
        Y_HAT = datapoint[1]
        
        # forward
        Y = model(X_HAT)
        if len(Y.shape) < 4:
            Y.squeeze(0)
        loss = loss_func.forward(Y, Y_HAT)
        
        # backward
        optim.zero_grad()
        loss.backward()
        
        # update
        optim.step()
        
        avg_loss += loss.item()
        loadingbar.set_postfix({
           "loss": avg_loss / (i+1)
        }, False)
        
    # print(Y)
    # print(Y_HAT)
    # print(get_bird(Y))
    # print(get_bird(Y_HAT))
    
    with torch.no_grad():
        avg_loss = 0
        avg_acc = 0
        loadingbar2 = tqdm(valid_data, ncols=80)
        model.eval()
        for i, datapoint in enumerate(loadingbar2):
            X_HAT = datapoint[0]
            Y_HAT = datapoint[1]
            # forward
            Y = model(X_HAT)
            if len(Y.shape) < 4:
                Y.squeeze(0)
            loss = loss_func.forward(Y, Y_HAT)
            acc = (torch.argmax(Y, 1) == torch.argmax(Y_HAT, 1)).float().mean()
            
            avg_loss += loss.item()
            avg_acc += acc.item()
            loadingbar2.set_postfix({
                "val_loss": avg_loss / (i+1),
                "val_acc": avg_acc / (i+1)
            }, False)
    print()
            
torch.save(model.state_dict(), get_ckpt_path(model_path_idx))

import os
from efficientnet_pytorch.model import EfficientNet
from efficientnet_pytorch.utils import efficientnet, efficientnet_params
import torch
import tqdm

def get_ckpt_path(i):
    return "./model" + str(i) + ".ckpt"

def get_ckpt_index():
    path = "./model"
    i = 0
    while os.path.exists(get_ckpt_path(i)):
        i += 1
    return i

def get_model(device, bird_order):
    width, depth, res, dropout = efficientnet_params('efficientnet-b0')
    blocks, params = efficientnet(1.4, 1.2, res, dropout, num_classes=len(bird_order))
    return EfficientNet(blocks, params).to(device)
    
def get_birds_out(device):
    with open('./birds.txt', 'r') as birdfile:
        birdtxt = birdfile.readlines()
        
    print("Creating bird out tensors")
    print("Getting names")
    birds_out_lists = []
    bird_names = set()
    for line in tqdm(birdtxt, ncols=80):
        line = line.strip()
        line_bird_names = line.split(' ')[1:]
        birds_out_lists.append(line_bird_names)
        for bird in line_bird_names:
            bird_names.add(bird)
            
    bird_order = list(sorted(bird_names))
    
    print("Creating tensors")
    birds_out = []
    for bird_list in tqdm(birds_out_lists, ncols=80):
        bird_tensor = [0] * len(bird_order)
        for i, bird_name in enumerate(bird_order):
            if bird_name in bird_list:
                bird_tensor[i] = 1
        birds_out.append(torch.Tensor(bird_tensor).to(device))
    
    return birds_out, bird_order

def get_bird(x, bird_order):
    dim = len(x.shape)-1
    x = torch.softmax(x, dim=dim)
    indices = torch.argmax(x, dim=dim)
        
    if len(indices.shape) >= 1:
        return [ bird_order[index] for index in indices]
    else:
        return bird_order[indices.item()]
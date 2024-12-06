import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

import torch
import torchvision
import torchvision.models as tvmodels
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import clip

import skimage.io as sio
from skimage import data, img_as_float
from skimage.transform import resize as imresize
from skimage.metrics import structural_similarity as ssim
import scipy as sp

import argparse
import time

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long', 'LLM'], required=True)
args = parser.parse_args()
cap_length = args.cap_length

os.environ['TORCH_HOME'] = './cache/torch_cache'  
os.environ['CLIP_CACHE_DIR'] = './cache/clip_cache'  

feats_dir = f'data/eval_features_{cap_length}_captions/subj01'
images_dir = f'results/versatile_diffusion_from_{cap_length}_captions/subj01'

if not os.path.exists(feats_dir):
   os.makedirs(feats_dir)

class batch_generator_external_images(Dataset):

    def __init__(self, data_path ='', prefix='', net_name='clip'):
        self.data_path = data_path
        self.prefix = prefix
        self.net_name = net_name
        
        if self.net_name == 'clip':
           self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        else:
           self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # if cap_length == 'preliminary':
        #     self.num_test = 200  # For preliminary, we only want images 800-999
        # else:
        self.num_test = 982  # Original number of test images
        
    def __getitem__(self,idx):
        img = Image.open('{}/{}{}.png'.format(self.data_path,self.prefix,idx))
        img = T.functional.resize(img,(224,224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return  self.num_test





global feat_list
feat_list = []
def fn(module, inputs, outputs):
    feat_list.append(outputs.cpu().numpy())

net_list = [
    ('inceptionv3','avgpool'),
    ('clip','final'),
    ('alexnet',2),
    ('alexnet',5),
    ('efficientnet','avgpool'),
    ('swav','avgpool')
    ]

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
net = None
batchsize=64

start_time = time.time()

for (net_name,layer) in net_list:
    feat_list = []
    print(net_name,layer)
    dataset = batch_generator_external_images(data_path=images_dir,net_name=net_name,prefix='')
    loader = DataLoader(dataset,batchsize,shuffle=False)
    
    if net_name == 'inceptionv3': # SD Brain uses this
        net = tvmodels.inception_v3(weights='IMAGENET1K_V1')
        if layer== 'avgpool':
            net.avgpool.register_forward_hook(fn) 
        elif layer == 'lastconv':
            net.Mixed_7c.register_forward_hook(fn)
            
    elif net_name == 'alexnet':
        net = tvmodels.alexnet(pretrained=True)
        if layer==2:
            net.features[4].register_forward_hook(fn)
        elif layer==5:
            net.features[11].register_forward_hook(fn)
        elif layer==7:
            net.classifier[5].register_forward_hook(fn)
            
    elif net_name == 'clip':
        model, _ = clip.load("ViT-L/14", download_root='./clip_cache', device=device)
        net = model.visual
        net = net.to(torch.float32)
        if layer==7:
            net.transformer.resblocks[7].register_forward_hook(fn)
        elif layer==12:
            net.transformer.resblocks[12].register_forward_hook(fn)
        elif layer=='final':
            net.register_forward_hook(fn)
    
    elif net_name == 'efficientnet':
        net = tvmodels.efficientnet_b1(weights=True)
        net.avgpool.register_forward_hook(fn) 
        
    elif net_name == 'swav':
        net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        net.avgpool.register_forward_hook(fn) 
    net.eval()
    net = net.to(device)    
    
    with torch.no_grad():
        for i,x in enumerate(loader):
            x = x.to(device)
            _ = net(x)
    if net_name == 'clip':
        if layer == 7 or layer == 12:
            feat_list = np.concatenate(feat_list,axis=1).transpose((1,0,2))
        else:
            feat_list = np.concatenate(feat_list)
    else:   
        feat_list = np.concatenate(feat_list)
    
    
    file_name = '{}/{}_{}.npy'.format(feats_dir,net_name,layer)
    np.save(file_name,feat_list)

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time for eval extract features {cap_length}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
print('='*50)
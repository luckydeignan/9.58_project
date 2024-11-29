import sys
sys.path.append('vdvae')
import torch
import numpy as np
import argparse
import os
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger, local_mpi_rank, mpi_size, maybe_download, mpi_rank
from data import mkdir_p
from vae import VAE
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
# import pdb
import time
start_time = time.time()

# Parse arguments
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-bs", "--bs", help="Batch Size", default=30)
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long'], required=True)
args = parser.parse_args()
cap_length = args.cap_length
batch_size = int(args.bs)

print('Loading VDVAE model...')

# VDVAE Hyperparameters
H = {'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500, 
     'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test',
     'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th',
     'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th',
     'restore_log_path': 'imagenet64-iter-1600000-log.jsonl',
     'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th',
     'dataset': 'imagenet64', 'ema_rate': 0.999,
     'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
     'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12',
     'zdim': 16, 'width': 512, 'bottleneck_multiple': 0.25, 'no_bias_above': 64,
     'num_mixtures': 10
     }

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)
ema_vae = load_vaes(H)

print('Loading predicted VDVAE features...')
# Load the predicted VDVAE features from the brain → caption → VDVAE pipeline
pred_latents = np.load(f'data/predicted_features/subj01/nsd_vdvae_from_{cap_length}_captions_pred_sub01_31l.npy')
# print("Loaded predicted latents shape:", pred_latents.shape)

# Load a single test image to get reference latent structure
image_path = 'data/processed_data/subj01/nsd_test_stim_sub1.npy'
test_images = np.load(image_path).astype(np.uint8)
single_image = test_images[0:1]  # Take first image
single_image = Image.fromarray(single_image[0])
single_image = T.functional.resize(single_image, (64,64))
single_image = torch.tensor(np.array(single_image)).float().unsqueeze(0)

# Get reference latent structure
data_input, _ = preprocess_fn(single_image)
with torch.no_grad():
    activations = ema_vae.encoder.forward(data_input)
    _, ref_latent = ema_vae.decoder.forward(activations, get_latents=True)

def latent_transformation(latents, ref):
    layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,
                          2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,
                          2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
                          2**12,2**12,2**14])
    transformed_latents = []
    for i in range(31):
        t_lat = latents[:,layer_dims[:i].sum():layer_dims[:i+1].sum()]
        c,h,w = ref[i]['z'].shape[1:]
        transformed_latents.append(t_lat.reshape(len(latents),c,h,w))
    return transformed_latents

def sample_from_hier_latents(latents, sample_ids):
    sample_ids = [id for id in sample_ids if id < len(latents[0])]
    layers_num = len(latents)
    sample_latents = []
    for i in range(layers_num):
        sample_latents.append(torch.tensor(latents[i][sample_ids]).float().cuda())
    return sample_latents

print('Transforming latents...')
input_latent = latent_transformation(pred_latents, ref_latent)
print("Transformed latents structure:")
for i, lat in enumerate(input_latent):
    print(f"Layer {i} shape:", lat.shape)

# Create output directory if it doesn't exist
os.makedirs(f'results/vdvae_from_{cap_length}_captions/subj01', exist_ok=True)

print('Generating images...')
for i in range(int(np.ceil(len(pred_latents)/batch_size))):
    print(f'Processing batch {i+1}/{int(np.ceil(len(pred_latents)/batch_size))}')
    samp = sample_from_hier_latents(input_latent, range(i*batch_size, (i+1)*batch_size))
    
    with torch.no_grad():
        px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
        sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
    
    for j in range(len(sample_from_latent)):
        im = sample_from_latent[j]
        im = Image.fromarray(im)
        im = im.resize((512,512), resample=3)
        im.save(f'results/vdvae_from_{cap_length}_captions/subj01/{i*batch_size+j}.png')

end_time = time.time()
duration = end_time - start_time
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)
print(f'Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}')
print(f'vdvae reconstruct images done for {cap_length} captions')

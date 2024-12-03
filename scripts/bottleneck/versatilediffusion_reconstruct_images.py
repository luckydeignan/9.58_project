import sys
sys.path.append('versatile_diffusion')
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean
import time

start_time = time.time()
os.environ['TRANSFORMERS_CACHE'] = './cache/huggingface_cache'

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long'], required=True)
parser.add_argument("-bs", "--bs", help="Batch Size", default=30)
parser.add_argument("-diff_str", "--diff_str", help="Diffusion Strength", default=0.75)
parser.add_argument("-mix_str", "--mix_str", help="Mixing Strength", default=0.4)
args = parser.parse_args()
cap_length = args.cap_length
batch_size = int(args.bs)
strength = float(args.diff_str)
mixing = float(args.mix_str)

def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'

    assert (x.shape[1]==512) & (x.shape[2]==512), 'Wrong image size'
    return x

print('Loading Versatile Diffusion model...')
cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

net.clip.cuda(0)
net.autokl.cuda(0)
net.autokl.half()
sampler = sampler(net)

print('Loading predicted features...')

pred_text = np.load(f'data/predicted_features/subj01/nsd_cliptext_from_{cap_length}_captions_pred_sub01.npy')
pred_text = torch.tensor(pred_text).half().cuda(1)
pred_vision = np.load(f'data/predicted_features/subj01/nsd_clipvision_from_{cap_length}_captions_pred_sub1.npy')
pred_vision = torch.tensor(pred_vision).half().cuda(1)

n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5

os.makedirs(f'results/versatile_diffusion_from_{cap_length}_captions/subj01', exist_ok=True)

print('Generating images...')
torch.manual_seed(0)

for i in range(int(np.ceil(len(pred_vision)/batch_size))):
    print(f'Processing batch {i+1}/{int(np.ceil(len(pred_vision)/batch_size))}')
    
    batch_start = i * batch_size
    batch_end = min((i + 1) * batch_size, len(pred_vision))
    
    for j in range(batch_start, batch_end):
        zim = Image.open(f'results/vdvae_from_{cap_length}_captions/subj01/{j}.png')

        zim = regularize_image(zim)
        zin = zim*2 - 1
        zin = zin.unsqueeze(0).cuda(0).half()

        init_latent = net.autokl_encode(zin)

        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        t_enc = int(strength * ddim_steps)
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to('cuda:0'))
        
        dummy_text = ''
        utx = net.clip_encode_text(dummy_text)
        utx = utx.cuda(1).half()
        
        dummy_image = torch.zeros((1,3,224,224)).cuda(0)
        uim = net.clip_encode_vision(dummy_image)
        uim = uim.cuda(1).half()
        
        z_enc = z_enc.cuda(1)

        cim = pred_vision[j].unsqueeze(0)
        ctx = pred_text[j].unsqueeze(0)
        
        sampler.model.model.diffusion_model.device='cuda:1'
        sampler.model.model.diffusion_model.half().cuda(1)
        
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, cim],
            second_conditioning=[utx, ctx],
            t_start=t_enc,
            unconditional_guidance_scale=scale,
            xtype='image', 
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=(1-mixing),
        )
        
        z = z.cuda(0).half()
        x = net.autokl_decode(z)
        color_adj = 'None'
        color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
        color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
        color_adj_keep_ratio = 0.5
        
        if color_adj_flag and (ctype=='vision'):
            x_adj = []
            for xi in x:
                color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
                xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
                x_adj.append(xi_adj)
            x = x_adj
        else:
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            x = [tvtrans.ToPILImage()(xi) for xi in x]
        save_path = f'results/versatile_diffusion_from_{cap_length}_captions/subj01/{j}.png'
        x[0].save(save_path)
        x[0].close()
        assert os.path.exists(save_path), f"Failed to save image {j}"
        sys.stdout.flush()
        print(f'Saved image {j}', flush=True)

end_time = time.time()
duration = end_time - start_time
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)
print(f'Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}')
print(f'versatile diffusion reconstruct images complete for {cap_length} captions')

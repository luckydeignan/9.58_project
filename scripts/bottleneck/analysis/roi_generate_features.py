import numpy as np
import pickle

import time
start_time = time.time()

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long', 'preliminary'], required=True)
args = parser.parse_args()
cap_length = args.cap_length

# Load ROI Masks

print("Loading ROI Masks")
with open(f'data/regression_weights/subj01/{cap_length}_caption_to_vdvae_regression_weights.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']
print(f"reg_w shape: {reg_w.shape}")
print(f"reg_b shape: {reg_b.shape}")

# Add this section to load train_latents
print("Loading VDVAE features")
nsd_features = np.load('data/extracted_features/subj01/nsd_vdvae_features_31l.npz')
if cap_length == 'preliminary':
    full_latents = nsd_features['train_latents']
    train_latents = full_latents[:800]  # Use first 800 samples for preliminary
else:
    train_latents = nsd_features['train_latents']
print(f"train_latents shape: {train_latents.shape}")

roi_dir = f'data/processed_data/subj01/roi_{cap_length}_captions'
num_rois = 13
roi_act = np.zeros((num_rois, 15724)).astype(np.float32)

# Load first ROI to check shape
temp_roi = np.load(f"{roi_dir}/floc-faces.npy")
print(f"Single ROI shape: {temp_roi.shape}")

# Load ROIs
roi_act[0] = np.load(f"{roi_dir}/floc-faces.npy")
roi_act[1] = np.load(f"{roi_dir}/floc-words.npy")
roi_act[2] = np.load(f"{roi_dir}/floc-places.npy")
roi_act[3] = np.load(f"{roi_dir}/floc-bodies.npy")
roi_act[4] = np.load(f"{roi_dir}/V1.npy")
roi_act[5] = np.load(f"{roi_dir}/V2.npy")
roi_act[6] = np.load(f"{roi_dir}/V3.npy")
roi_act[7] = np.load(f"{roi_dir}/V4.npy")
roi_act[8] = np.load(f"{roi_dir}/ecc05.npy")
roi_act[9] = np.load(f"{roi_dir}/ecc10.npy")
roi_act[10] = np.load(f"{roi_dir}/ecc20.npy")
roi_act[11] = np.load(f"{roi_dir}/ecc40.npy")
roi_act[12] = np.load(f"{roi_dir}/ecc40p.npy")

roi_act[roi_act>0]=1
roi_act[roi_act<0]=0
print(f"roi_act shape after loading: {roi_act.shape}")

# Load ROI to caption weights
with open(f'data/regression_weights/subj01/{cap_length}_roi_to_caption_weights.pkl',"rb") as f:
    roi_datadict = pickle.load(f)
    roi_w = roi_datadict['weight']
    roi_b = roi_datadict['bias']
print(f"roi_w shape: {roi_w.shape}")
print(f"roi_b shape: {roi_b.shape}")

# First map ROI to caption space
caption_space = roi_act @ roi_w.T + roi_b
print(f"caption_space shape: {caption_space.shape}")

# Then map to VDVAE space using existing weights
pred_vae = caption_space @ reg_w.T
print(f"initial pred_vae shape: {pred_vae.shape}")

# Normalize
pred_vae = pred_vae / (np.linalg.norm(pred_vae,axis=1).reshape((num_rois,1)) + 1e-8)
pred_vae = pred_vae * 50 + reg_b

pred_vae = (pred_vae - np.mean(pred_vae,axis=0)) / np.std(pred_vae,axis=0)
print(f"pred_vae after normalization shape: {pred_vae.shape}")

pred_vae = pred_vae * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
pred_vae = pred_vae / np.linalg.norm(pred_vae,axis=1).reshape((num_rois,1))
pred_vae = pred_vae * 80
print(f"final pred_vae shape: {pred_vae.shape}")
np.save(f'data/predicted_features/subj01/nsd_vdvae_from_{cap_length}_captions_pred_sub01_31l.npy',pred_vae)

# Generate CLIP-Text Features

with open(f'data/regression_weights/subj01/{cap_length}_caption_to_cliptext_regression_weights.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

# Use caption_space instead of roi_act directly
pred_clipt = np.zeros((num_rois,reg_w.shape[0],reg_w.shape[1])).astype(np.float32)
for j in range(reg_w.shape[0]):
    pred_clipt[:,j] = (caption_space @ reg_w[j].T) 
    
pred_clipt = pred_clipt / (np.linalg.norm(pred_clipt,axis=(1,2)).reshape((num_rois,1,1)) + 1e-8)
pred_clipt = pred_clipt * 9 + reg_b

np.save(f'data/predicted_features/subj01/nsd_cliptext_{cap_length}_roi_nsdgeneral.npy',pred_clipt)

# Generate CLIP-Vision Features

with open(f'data/regression_weights/subj01/{cap_length}_caption_to_clipvision_regression_weights.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']
    
pred_clipv = np.zeros((num_rois,reg_w.shape[0],reg_w.shape[1])).astype(np.float32)
for j in range(reg_w.shape[0]):
    pred_clipv[:,j] = (caption_space @ reg_w[j].T) 
    
pred_clipv = pred_clipv / (np.linalg.norm(pred_clipv,axis=(1,2)).reshape((num_rois,1,1)) + 1e-8)
pred_clipv = pred_clipv * 15 + reg_b

np.save(f'data/predicted_features/subj01/nsd_clipvision_{cap_length}_roi_nsdgeneral.npy',pred_clipv)

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time for ROI feature generation {cap_length}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
print('='*50)
print('  ')
print('  ')
print('  ')
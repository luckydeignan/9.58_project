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

with open(f'data/regression_weights/subj01/{cap_length}_caption_to_vdvae_regression_weights.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

roi_dir = f'data/processed_data/subj01/roi_{cap_length}_captions'
num_rois = 13
roi_act = np.zeros((num_rois, reg_w.shape[1])).astype(np.float32)

roi_act[0] = np.load("{}/floc-faces.npy".format(roi_dir))
roi_act[1] = np.load("{}/floc-words.npy".format(roi_dir))
roi_act[2] = np.load("{}/floc-places.npy".format(roi_dir))
roi_act[3] = np.load("{}/floc-bodies.npy".format(roi_dir))
roi_act[4] = np.load("{}/V1.npy".format(roi_dir))
roi_act[5] = np.load("{}/V2.npy".format(roi_dir))
roi_act[6] = np.load("{}/V3.npy".format(roi_dir))
roi_act[7] = np.load("{}/V4.npy".format(roi_dir))
roi_act[8] = np.load("{}/ecc05.npy".format(roi_dir))
roi_act[9] = np.load("{}/ecc10.npy".format(roi_dir))
roi_act[10] = np.load("{}/ecc20.npy".format(roi_dir))
roi_act[11] = np.load("{}/ecc40.npy".format(roi_dir))
roi_act[12] = np.load("{}/ecc40p.npy".format(roi_dir))


roi_act[roi_act>0]=1
roi_act[roi_act<0]=0

# Generate VDVAE Features
if cap_length == 'preliminary':
    nsd_features = np.load('data/extracted_features/subj01/nsd_vdvae_features_31l.npz')
    full_latents = nsd_features['train_latents']
    train_latents = full_latents[:800]
else:
    nsd_features = np.load('data/extracted_features/subj01/nsd_vdvae_features_31l.npz')
    train_latents = nsd_features['train_latents']

pred_vae = (roi_act @ reg_w.T) 
pred_vae = pred_vae / (np.linalg.norm(pred_vae,axis=1).reshape((num_rois,1)) + 1e-8)
pred_vae = pred_vae * 50 + reg_b

pred_vae = (pred_vae - np.mean(pred_vae,axis=0)) / np.std(pred_vae,axis=0)
pred_vae = pred_vae * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
pred_vae = pred_vae / np.linalg.norm(pred_vae,axis=1).reshape((num_rois,1))
pred_vae = pred_vae * 80
np.save(f'data/predicted_features/subj01/nsd_vdvae_from_{cap_length}_captions_pred_sub01_31l.npy',pred_vae)

# Generate CLIP-Text Features

with open(f'data/regression_weights/subj01/{cap_length}_caption_to_cliptext_regression_weights.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

pred_clipt = np.zeros((num_rois,reg_w.shape[0],reg_w.shape[1])).astype(np.float32)
for j in range(reg_w.shape[0]):
    pred_clipt[:,j] = (roi_act @ reg_w[j].T) 
    
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
    pred_clipv[:,j] = (roi_act @ reg_w[j].T) 
    
pred_clipv = pred_clipv / (np.linalg.norm(pred_clipv,axis=(1,2)).reshape((num_rois,1,1)) + 1e-8)
pred_clipv = pred_clipv * 15 + reg_b

np.save(f'data/predicted_features/subj01/nsd_clipvision_{cap_length}_roi_nsdgeneral.npy',pred_clipv)

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time for evaluate reconstruction {cap_length}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
print('='*50)
print('  ')
print('  ')
print('  ')
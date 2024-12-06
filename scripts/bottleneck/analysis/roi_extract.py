import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

import time
start_time = time.time()

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long', 'LLM'], required=True)
args = parser.parse_args()
cap_length = args.cap_length


roi_dir = 'data/nsddata/ppdata/subj01/func1pt8mm/roi/'
betas_dir = 'data/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/'
res_roi_dir = f'data/processed_data/subj01/roi_{cap_length}_captions/'
if not os.path.exists(res_roi_dir):
   os.makedirs(res_roi_dir)

nsdgeneral_mask_filename = 'nsdgeneral.nii.gz'
nsdgeneral_mask = nib.load(roi_dir+nsdgeneral_mask_filename).get_fdata()
nsdgeneral_mask[nsdgeneral_mask<0] = 0
num_voxel = nsdgeneral_mask[nsdgeneral_mask>0].shape[0]
print(f'NSD General : {num_voxel}')


mask_files = [
              'floc-faces.nii.gz',
              'floc-words.nii.gz',
              'floc-places.nii.gz',
              'floc-bodies.nii.gz'
              ]


    
for mfile in mask_files:
    roi_mask = nib.load(roi_dir+mfile).get_fdata()
    np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/{mfile[:-7]}.npy', roi_mask[nsdgeneral_mask>0])
    

roi_mask = nib.load(roi_dir+mask_files[0]).get_fdata()
v1 = np.zeros_like(nsdgeneral_mask)
v2 = np.zeros_like(nsdgeneral_mask)
v3 = np.zeros_like(nsdgeneral_mask)
v4 = np.zeros_like(nsdgeneral_mask)

v1[roi_mask==1] = 1
v1[roi_mask==2] = 1
v2[roi_mask==3] = 1
v2[roi_mask==4] = 1
v3[roi_mask==5] = 1
v3[roi_mask==6] = 1
v4[roi_mask==7] = 1

np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/V1.npy', v1[nsdgeneral_mask>0])
np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/V2.npy', v2[nsdgeneral_mask>0])
np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/V3.npy', v3[nsdgeneral_mask>0])
np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/V4.npy', v4[nsdgeneral_mask>0])


roi_mask = nib.load(roi_dir+"prf-eccrois.nii.gz").get_fdata()
ecc05 = np.zeros_like(nsdgeneral_mask)
ecc10 = np.zeros_like(nsdgeneral_mask)
ecc20 = np.zeros_like(nsdgeneral_mask)
ecc40 = np.zeros_like(nsdgeneral_mask)
ecc40p = np.zeros_like(nsdgeneral_mask)

ecc05[roi_mask==1] = 1
ecc10[roi_mask==2] = 1
ecc20[roi_mask==3] = 1
ecc40[roi_mask==4] = 1
ecc40p[roi_mask==5] = 1

np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/ecc05.npy', ecc05[nsdgeneral_mask>0])
np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/ecc10.npy', ecc10[nsdgeneral_mask>0])
np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/ecc20.npy', ecc20[nsdgeneral_mask>0])
np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/ecc40.npy', ecc40[nsdgeneral_mask>0])
np.save(f'data/processed_data/subj01/roi_{cap_length}_captions/ecc40p.npy', ecc40p[nsdgeneral_mask>0])

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time for evaluate reconstruction {cap_length}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
print('='*50)
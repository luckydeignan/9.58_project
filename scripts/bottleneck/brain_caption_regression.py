# fMRI --> caption --> vdvae, CLIP text, CLIP vision
# need to train brain to caption --> save file in npy file; number of rows is test input; number of columns is dimension of caption embeddings
# goal is to use predicted captions from fMRI to predict VDVAE, CLIP text & vision

# follow and modify existing regression scripts
# need a regression from text-embeddings to VDVAE, CLIP text & vision

import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
# import pdb

# parse arguments
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long'], required=True)
args = parser.parse_args()
cap_length = args.cap_length

# Load caption embeddings
print("Loading caption embeddings")
train_captions = np.load(f'data/caption_embeddings/subj01/{cap_length}er_truncated_caption_bottleneck_embeddings_sub1.npy')
# print("Caption embeddings shape:", train_captions.shape)
# pdb.set_trace()  # Verify caption embeddings loaded correctly

# Load fMRI data
print("Loading fMRI data")
train_path = 'data/processed_data/subj01/nsd_train_fmriavg_nsdgeneral_sub1.npy'
train_fmri = np.load(train_path)
test_path = 'data/processed_data/subj01/nsd_test_fmriavg_nsdgeneral_sub1.npy'
test_fmri = np.load(test_path)
# print("fMRI shapes - train:", train_fmri.shape, "test:", test_fmri.shape)
# pdb.set_trace()  # Verify fMRI data loaded correctly

# Initial normalization of fMRI data (following vdvae_regression.py)
train_fmri = train_fmri/300
test_fmri = test_fmri/300

# Standardize fMRI data
norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

# print("\nfMRI Statistics after normalization:")
# print("Train - mean:", np.mean(train_fmri), "std:", np.std(train_fmri))
# print("Test  - mean:", np.mean(test_fmri), "std:", np.std(test_fmri))
# print("Train - max:", np.max(train_fmri), "min:", np.min(train_fmri))
# print("Test  - max:", np.max(test_fmri), "min:", np.min(test_fmri))
# pdb.set_trace()  # Verify normalization worked correctly

# num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)
# print("\nDimensions:")
# print("Number of voxels:", num_voxels)
# print("Number of training samples:", num_train)
# print("Number of test samples:", num_test)
# pdb.set_trace()  # Verify dimensions make sense

## Caption Embeddings Regression
print('Training Caption Embeddings Regression')
reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.fit(train_fmri, train_captions)

# Quick check of initial predictions
pred_test_sample = reg.predict(test_fmri[:5])
# print("\nSample prediction shape:", pred_test_sample.shape)
# print("Sample prediction stats - mean:", np.mean(pred_test_sample), "std:", np.std(pred_test_sample))
# pdb.set_trace()  # Verify initial predictions look reasonable

# Full predictions
pred_test_captions = reg.predict(test_fmri)
std_norm_test_captions = (pred_test_captions - np.mean(pred_test_captions, axis=0)) / np.std(pred_test_captions, axis=0)
pred_captions = std_norm_test_captions * np.std(train_captions, axis=0) + np.mean(train_captions, axis=0)

print("Final prediction shapes and stats:")
# print("Prediction shape:", pred_captions.shape)
# print("Prediction - mean:", np.mean(pred_captions), "std:", np.std(pred_captions))
# print("Original captions - mean:", np.mean(train_captions), "std:", np.std(train_captions))
# pdb.set_trace()  # Verify final predictions look reasonable

# Save predictions
np.save(f'data/predicted_features/subj01/nsd_{cap_length}_captions_predtest_nsdgeneral.npy', pred_captions)

# Save regression weights
datadict = {
    'weight': reg.coef_,
    'bias': reg.intercept_,
}

with open(f'data/regression_weights/subj01/{cap_length}_caption_regression_weights.pkl', "wb") as f:
    pickle.dump(datadict, f)

print(f'brain caption regression complete for {cap_length} captions')
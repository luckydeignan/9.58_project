import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long'], required=True)
args = parser.parse_args()
cap_length = args.cap_length

print("Loading predicted captions")
pred_captions = np.load(f'data/predicted_features/subj01/nsd_{cap_length}_captions_predtest_nsdgeneral.npy')

print("Loading training captions")
train_captions = np.load(f'data/caption_embeddings/subj01/{cap_length}er_truncated_caption_bottleneck_embeddings_sub1.npy')

print("Loading VDVAE features")
nsd_features = np.load(f'data/extracted_features/subj01/nsd_vdvae_features_31l.npz')
train_latents = nsd_features['train_latents']

print('Training Caption to VDVAE Regression')
reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.fit(train_captions, train_latents)

pred_sample = reg.predict(pred_captions[:5])

pred_vdvae = reg.predict(pred_captions)
std_norm_pred_vdvae = (pred_vdvae - np.mean(pred_vdvae, axis=0)) / np.std(pred_vdvae, axis=0)
final_pred_vdvae = std_norm_pred_vdvae * np.std(train_latents, axis=0) + np.mean(train_latents, axis=0)

print("Prediction shapes and stats:")

std_norm_pred_vdvae = (pred_vdvae - np.mean(pred_vdvae, axis=0)) / np.std(pred_vdvae, axis=0)
final_pred_vdvae = std_norm_pred_vdvae * np.std(train_latents, axis=0) + np.mean(train_latents, axis=0)

save_path = f'data/predicted_features/subj01/nsd_vdvae_from_{cap_length}_captions_pred_sub01_31l.npy'
np.save(save_path, final_pred_vdvae)

datadict = {
    'weight': reg.coef_,
    'bias': reg.intercept_,
}

with open(f'data/regression_weights/subj01/{cap_length}_caption_to_vdvae_regression_weights.pkl', "wb") as f:
    pickle.dump(datadict, f)

print(f'caption to vdvae regression complete for {cap_length} captions')
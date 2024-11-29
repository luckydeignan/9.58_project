import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
# import pdb

# Parse arguments
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long'], required=True)
args = parser.parse_args()
cap_length = args.cap_length

# Load predicted captions (from brain_caption_regression.py output)
print("Loading predicted captions")
pred_captions = np.load(f'data/predicted_features/subj01/nsd_{cap_length}_captions_predtest_nsdgeneral.npy')
# print("Predicted captions shape:", pred_captions.shape)
# pdb.set_trace()  # Verify predicted captions loaded correctly

# Load training captions
print("Loading training captions")
train_captions = np.load(f'data/caption_embeddings/subj01/{cap_length}er_truncated_caption_bottleneck_embeddings_sub1.npy')
# print("Training captions shape:", train_captions.shape)
# pdb.set_trace()  # Verify training captions loaded correctly

# Load VDVAE features (training set for establishing normalization parameters)
print("Loading VDVAE features")
nsd_features = np.load(f'data/extracted_features/subj01/nsd_vdvae_features_31l.npz')
train_latents = nsd_features['train_latents']
# print("VDVAE training latents shape:", train_latents.shape)
# pdb.set_trace()  # Verify VDVAE features loaded correctly

# Train caption to VDVAE regression
print('Training Caption to VDVAE Regression')
reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.fit(train_captions, train_latents)

# Quick check of initial predictions
pred_sample = reg.predict(pred_captions[:5])
# print("Sample prediction shape:", pred_sample.shape)
# print("Sample prediction stats - mean:", np.mean(pred_sample), "std:", np.std(pred_sample))
# pdb.set_trace()  # Verify initial predictions look reasonable

# Generate predictions
pred_vdvae = reg.predict(pred_captions)
std_norm_pred_vdvae = (pred_vdvae - np.mean(pred_vdvae, axis=0)) / np.std(pred_vdvae, axis=0)
final_pred_vdvae = std_norm_pred_vdvae * np.std(train_latents, axis=0) + np.mean(train_latents, axis=0)

print("Prediction shapes and stats:")
# print("VDVAE prediction shape:", final_pred_vdvae.shape)
# print("Prediction - mean:", np.mean(final_pred_vdvae), "std:", np.std(final_pred_vdvae))
# print("Original VDVAE - mean:", np.mean(train_latents), "std:", np.std(train_latents))
# pdb.set_trace()  # Verify final predictions look reasonable

# Normalize predictions using training statistics
std_norm_pred_vdvae = (pred_vdvae - np.mean(pred_vdvae, axis=0)) / np.std(pred_vdvae, axis=0)
final_pred_vdvae = std_norm_pred_vdvae * np.std(train_latents, axis=0) + np.mean(train_latents, axis=0)

# print("Prediction shapes and stats:")
# print("VDVAE prediction shape:", final_pred_vdvae.shape)
# print("Prediction - mean:", np.mean(final_pred_vdvae), "std:", np.std(final_pred_vdvae))
# print("Original VDVAE - mean:", np.mean(train_latents), "std:", np.std(train_latents))

# Save predictions
save_path = f'data/predicted_features/subj01/nsd_vdvae_from_{cap_length}_captions_pred_sub01_31l.npy'
np.save(save_path, final_pred_vdvae)

# Save regression weights
datadict = {
    'weight': reg.coef_,
    'bias': reg.intercept_,
}

with open(f'data/regression_weights/subj01/{cap_length}_caption_to_vdvae_regression_weights.pkl', "wb") as f:
    pickle.dump(datadict, f)

print(f'caption to vdvae regression complete for {cap_length} captions')
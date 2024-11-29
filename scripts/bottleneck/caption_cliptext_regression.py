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

# Load CLIP text features
print("Loading CLIP text features")
train_clip = np.load(f'data/extracted_features/subj01/nsd_cliptext_train.npy')
# print("CLIP text features shape:", train_clip.shape)
# pdb.set_trace()  # Verify CLIP features loaded correctly

# Train caption to CLIP text regression
print('Training Caption to CLIP Text Regression')
num_samples, num_embed, num_dim = train_clip.shape
reg_w = np.zeros((num_embed, num_dim, train_captions.shape[1])).astype(np.float32)
reg_b = np.zeros((num_embed, num_dim)).astype(np.float32)
pred_clip = np.zeros((len(pred_captions), num_embed, num_dim))

for i in range(num_embed):
    print(f"Training embedding {i}/{num_embed}")
    reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True)
    reg.fit(train_captions, train_clip[:,i])
    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(pred_captions)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent, axis=0)) / np.std(pred_test_latent, axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i], axis=0) + np.mean(train_clip[:,i], axis=0)

print("Prediction shapes and stats:")
# print("CLIP text prediction shape:", pred_clip.shape)
# print("Prediction - mean:", np.mean(pred_clip), "std:", np.std(pred_clip))
# print("Original CLIP - mean:", np.mean(train_clip), "std:", np.std(train_clip))
# pdb.set_trace()  # Verify final predictions look reasonable

# Save predictions
save_path = f'data/predicted_features/subj01/nsd_cliptext_from_{cap_length}_captions_pred_sub01.npy'
np.save(save_path, pred_clip)

# Save regression weights
datadict = {
    'weight': reg_w,
    'bias': reg_b,
}

with open(f'data/regression_weights/subj01/{cap_length}_caption_to_cliptext_regression_weights.pkl', "wb") as f:
    pickle.dump(datadict, f)

print(f'caption to cliptext regression done for {cap_length} captions')

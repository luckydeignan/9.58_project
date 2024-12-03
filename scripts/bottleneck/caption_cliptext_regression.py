import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long', 'preliminary'], required=True)
args = parser.parse_args()
cap_length = args.cap_length

print("Loading predicted captions")
pred_captions = np.load(f'data/predicted_features/subj01/nsd_{cap_length}_captions_predtest_nsdgeneral.npy')

print("Loading training captions")
if (cap_length == 'preliminary'):
    train_captions = np.load(f'data/caption_embeddings/subj01/preliminary_TRAIN_LLM_caption_bottleneck_embeddings_sub1.npy')
else:
    train_captions = np.load(f'data/caption_embeddings/subj01/{cap_length}er_truncated_caption_bottleneck_embeddings_sub1.npy')

print("Loading CLIP text features")
if (cap_length == 'preliminary'):
    clip_features = np.load(f'data/extracted_features/subj01/nsd_cliptext_train.npy')
    train_clip = clip_features[:800]
else:
    train_clip = np.load(f'data/extracted_features/subj01/nsd_cliptext_train.npy')

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

save_path = f'data/predicted_features/subj01/nsd_cliptext_from_{cap_length}_captions_pred_sub01.npy'
np.save(save_path, pred_clip)

datadict = {
    'weight': reg_w,
    'bias': reg_b,
}

with open(f'data/regression_weights/subj01/{cap_length}_caption_to_cliptext_regression_weights.pkl', "wb") as f:
    pickle.dump(datadict, f)

print(f'caption to cliptext regression done for {cap_length} captions')

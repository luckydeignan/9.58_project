import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long', 'LLM'], required=True)
args = parser.parse_args()
cap_length = args.cap_length

print("Loading caption embeddings")

print("Loading fMRI data")

if (cap_length == 'LLM'):
    # for LLM data
    train_captions = np.load(f'data/caption_embeddings/subj01/final_LLM_caption_bottleneck_embeddings_sub1.npy')
else:
    train_captions = np.load(f'data/caption_embeddings/subj01/{cap_length}er_truncated_caption_bottleneck_embeddings_sub1.npy')

train_path = 'data/processed_data/subj01/nsd_train_fmriavg_nsdgeneral_sub1.npy'
train_fmri = np.load(train_path)
test_path = 'data/processed_data/subj01/nsd_test_fmriavg_nsdgeneral_sub1.npy'
test_fmri = np.load(test_path)

norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print('Training Caption Embeddings Regression')
reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.fit(train_fmri, train_captions)

pred_test_sample = reg.predict(test_fmri[:5])

pred_test_captions = reg.predict(test_fmri)
std_norm_test_captions = (pred_test_captions - np.mean(pred_test_captions, axis=0)) / np.std(pred_test_captions, axis=0)
pred_captions = std_norm_test_captions * np.std(train_captions, axis=0) + np.mean(train_captions, axis=0)

print("Final prediction shapes and stats:")

np.save(f'data/predicted_features/subj01/nsd_{cap_length}_captions_predtest_nsdgeneral.npy', pred_captions)

datadict = {
    'weight': reg.coef_,
    'bias': reg.intercept_,
}

with open(f'data/regression_weights/subj01/{cap_length}_caption_regression_weights.pkl', "wb") as f:
    pickle.dump(datadict, f)

print(f'brain caption regression complete for {cap_length} captions')
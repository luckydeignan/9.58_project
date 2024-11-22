# fMRI --> caption --> vdvae, CLIP text, CLIP vision
# need to train brain to caption --> save file in npy file; number of rows is test input; number of columns is dimension of caption embeddings
# goal is to use predicted captions from fMRI to predict VDVAE, CLIP text & vision

# follow and modify existing regression scripts
# need a regression from text-embeddings to VDVAE, CLIP text & vision

import numpy as np
import sklearn.linear_model as skl
import pickle #??
import argpase #??

# parse arguments
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
sub = int(args.sub)
assert sub in [1,2,5,7]

# load fmri data
print("Loading fMRI data")
train_fmri = np.load('data/processed_data/subj{:02d}/nsd_train_fmri_sub{}.npy'.format(sub, sub))
test_fmri = np.load('data/processed_data/subj{:02d}/nsd_test_fmri_sub{}.npy'.format(sub, sub))

# Normalize
train_fmri = (train_fmri - np.mean(train_fmri, axis=0)) / np.std(train_fmri, axis=0)
test_fmri = (test_fmri - np.mean(test_fmri, axis=0)) / np.std(test_fmri, axis=0)

# load caption embeddings
print("Loading caption embeddings")
train_captions = np.load('data/processed_data/subj{:02d}/nsd_train_cap_embeddings_sub{}.npy'.format(sub, sub))
test_captions = np.load('data/processed_data/subj{:02d}/nsd_test_cap_embeddings_sub{}.npy'.format(sub, sub))

num_voxels = train_fmri.shape[1]
num_samples, embedding_dim = train_captions.shape

print("Training Regresison")
reg_w = np.zeros((embedding_dim, num_voxels)).astype(np.float32)
reg_b = np.zeros(embedding_dim).astype(np.float32)
pred_captions = np.zeros_like(test_captions)

# train regression model
reg = skl.Ridge(alpha=100000,, max_iter=50000, fit_intercept=True)
reg.fit(train_fmrim train_captions)
reg_w = reg.coef_
reg_b = reg.intercept_

# predictions
pred_test = reg.predict(test_fmri)
# normalize 
std_norm_test = (pred_test - np.mean(pred_test, axis=0)) np.std(pred_test, axis=0)
pred_captions = std_norm_test * np.std(train_captions, axis=0) + np.mean(train_captions, axis=0)

print("R2 Score:", reg.score(test_fmri, test_captions))

# save predictions
print("Saving predictions and weights")
np.save('data/predicted_features/subj{:02d}/nsd_captions_predtest_nsdgeneral.npy'.format(sub), pred_captions)

# save weights
data_dict = {
    'weight':reg_w,
    'bias': reg_b,
}

with open('data/reression_weights/subj{:02d}/caption_regression_weights.pkl'.format(sub), 'wb') as f:
    pickle.dump(data_dict, f)

print('done')
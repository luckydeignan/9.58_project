import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

sub = 1
directory = 'data/caption_embeddings/subj{:02d}'.format(sub)


train_caps = list(np.load('{}/preliminary_raw_train_caps_sub{}.npy'.format(directory,sub)))
test_caps = list(np.load('{}/preliminary_raw_test_caps_sub{}.npy'.format(directory,sub)))

import pdb; pdb.set_trace()
model = SentenceTransformer("all-MiniLM-L6-v2")

train_embeddings = model.encode(train_caps)
test_embeddings = model.encode(test_caps)

# Apply PCA to reduce to 50 dimensions
pca = PCA(n_components=50)
train_reduced_embeddings = pca.fit_transform(train_embeddings)
test_reduced_embeddings = pca.fit_transform(test_embeddings)

# Output shape will now be (num_sentences, 50)
print("Train Original shape:", train_embeddings.shape)          # (num_sentences, 384)
print("Train Reduced shape:", train_reduced_embeddings.shape)    # (num_sentences, 50)
print("Test Original shape:", test_embeddings.shape)          # (num_sentences, 384)
print("Test Reduced shape:", test_reduced_embeddings.shape)    # (num_sentences, 50)

np.save('{}/preliminary_TRAIN_LLM_caption_bottleneck_embeddings_sub{}.npy'.format(directory,sub), train_reduced_embeddings)
np.save('{}/preliminary_TEST_LLM_caption_bottleneck_embeddings_sub{}.npy'.format(directory,sub), test_reduced_embeddings)


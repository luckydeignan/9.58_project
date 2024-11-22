import numpy as np

sub = 1
embeddings = np.load('data/caption_embeddings/subj{:02d}/truncated_caption_bottleneck_embeddings_sub{}.npy'.format(sub,sub))

print(len(embeddings))

for i in range(2):
    print(len(embeddings[i]))
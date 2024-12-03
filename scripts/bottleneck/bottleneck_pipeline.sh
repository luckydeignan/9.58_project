#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cap_length>"
    echo "cap_length must be either 'short' or 'long'"
    exit 1
fi

CAP_LENGTH=$1

if [ "$CAP_LENGTH" != "short" ] && [ "$CAP_LENGTH" != "long" ]; then
    echo "Error: cap_length must be either 'short' or 'long'"
    exit 1
fi

echo "Starting pipeline with cap_length: $CAP_LENGTH"

python scripts/bottleneck/brain_caption_regression.py -cap $CAP_LENGTH && \
python scripts/bottleneck/caption_vdvae_regression.py -cap $CAP_LENGTH && \
python scripts/bottleneck/vdvae_reconstruct_images.py -cap $CAP_LENGTH && \
python scripts/bottleneck/caption_cliptext_regression.py -cap $CAP_LENGTH && \
python scripts/bottleneck/caption_clipvision_regression.py -cap $CAP_LENGTH && \
python scripts/bottleneck/versatilediffusion_reconstruct_images.py -cap $CAP_LENGTH

if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline failed!"
    exit 1
fi
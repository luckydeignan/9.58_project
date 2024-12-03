#!/bin/bash

set -e  # Exit on any error
set -u  # Exit on undefined variable

echo "Starting bottleneck evaluation pipeline"

python scripts/bottleneck/analysis/eval_extract_features.py -cap long && \
python scripts/bottleneck/analysis/evaluate_reconstruction.py -cap long && \
python scripts/bottleneck/analysis/eval_extract_features.py -cap short && \
python scripts/bottleneck/analysis/evaluate_reconstruction.py -cap short

if [ $? -eq 0 ]; then
    echo "Bottleneck evaluation complete successfully!"
else
    echo "Bottleneck evaluation pipeline failed!"
    exit 1
fi

echo "Now starting ROI analysis pipeline"

python scripts/bottleneck/analysis/roi_extract.py -cap long && \
python scripts/bottleneck/analysis/roi_generate_features.py -cap long && \
python scripts/bottleneck/analysis/roi_vdvae_reconstruction.py -cap long && \
python scripts/bottleneck/analysis/roi_versatilediffusion_reconstruct.py -cap long && \
python scripts/bottleneck/analysis/roi_extract.py -cap short && \
python scripts/bottleneck/analysis/roi_generate_features.py -cap short && \
python scripts/bottleneck/analysis/roi_vdvae_reconstruction.py -cap short && \
python scripts/bottleneck/analysis/roi_versatilediffusion_reconstruct.py -cap short

if [ $? -eq 0 ]; then
    echo "Bottleneck ROI complete successfully!"
else
    echo "Bottleneck ROI pipeline failed!"
    exit 1
fi
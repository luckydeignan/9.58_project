#!/bin/bash

set -e  
set -u 

echo "Starting LLM regression pipeline"

python scripts/bottleneck/brain_caption_regression.py -cap LLM
python scripts/bottleneck/caption_vdvae_regression.py -cap LLM
python scripts/bottleneck/vdvae_reconstruct_images.py -cap LLM
python scripts/bottleneck/caption_cliptext_regression.py -cap LLM
python scripts/bottleneck/caption_clipvision_regression.py -cap LLM
python scripts/bottleneck/versatilediffusion_reconstruct_images.py -cap LLM

echo "regression + reconstruction completed successfully!"

echo "Starting LLM evaluation & ROI pipeline"  

python scripts/bottleneck/analysis/eval_extract_features.py -cap LLM
python scripts/bottleneck/analysis/evaluate_reconstruction.py -cap LLM
python scripts/bottleneck/analysis/roi_extract.py -cap LLM
python scripts/bottleneck/analysis/roi_caption_regression.py -cap LLM
python scripts/bottleneck/analysis/roi_generate_features.py -cap LLM
python scripts/bottleneck/analysis/roi_vdvae_reconstruct.py -cap LLM
python scripts/bottleneck/analysis/roi_versatilediffusion_reconstruct.py -cap LLM

echo "Bottleneck eval & ROI complete successfully!"

python scripts/bottleneck/analysis/visuals.py
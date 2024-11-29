import subprocess
import time
import argparse
import sys
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"Starting: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    try:
        subprocess.run(command, check=True)
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        print(f"\nCompleted: {description}")
        print(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    except subprocess.CalledProcessError as e:
        print(f"\nError in {description}")
        print(f"Error details: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run bottleneck pipeline')
    parser.add_argument('-cap', '--cap_length', choices=['short', 'long'], required=True,
                       help='Caption length (short/long)')
    parser.add_argument('-bs', '--batch_size', type=int, default=30,
                       help='Batch size for image generation')
    parser.add_argument('-diff_str', '--diff_strength', type=float, default=0.75,
                       help='Diffusion strength for VD')
    parser.add_argument('-mix_str', '--mix_strength', type=float, default=0.4,
                       help='Mixing strength for VD')
    
    args = parser.parse_args()
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    
    pipeline_steps = [
        {
            "script": "brain_caption_regression.py",
            "description": "Brain to Caption Regression",
            "args": ["-cap", args.cap_length]
        },
        {
            "script": "caption_vdvae_regression.py",
            "description": "Caption to VDVAE Regression",
            "args": ["-cap", args.cap_length]
        },
        {
            "script": "vdvae_reconstruct_images.py",
            "description": "VDVAE Image Reconstruction",
            "args": ["-cap", args.cap_length]
        },
        {
            "script": "caption_cliptext_regression.py",
            "description": "Caption to CLIP Text Regression",
            "args": ["-cap", args.cap_length]
        },
        {
            "script": "caption_clipvision_regression.py",
            "description": "Caption to CLIP Vision Regression",
            "args": ["-cap", args.cap_length]
        },
        {
            "script": "versatilediffusion_reconstruct_images.py",
            "description": "Versatile Diffusion Image Reconstruction",
            "args": ["-cap", args.cap_length]
        }
    ]

    total_start_time = time.time()
    
    for step in pipeline_steps:
        command = [sys.executable, str(script_dir / step["script"])] + step["args"]
        run_command(command, step["description"])
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    print(f"\n{'='*50}")
    print(f"Complete Pipeline Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

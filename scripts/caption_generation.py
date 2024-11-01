"""
File that will take in image as input, fetch LLM API, and generate captions of varying length of said image

Input: PNG image
Output: caption
"""
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import base64
from io import BytesIO
from openai import OpenAI
import os
import google.generativeai as genai

class ImageLoader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img, (64, 64))
        img = torch.tensor(np.array(img)).float()
        return img

    def __len__(self):
        return len(self.im)

# Example usage:
sub = 1
image_path = 'data/processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub)
dataset = ImageLoader(data_path=image_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

def tensor_to_pil(tensor_image):
    """Convert a PyTorch tensor to PIL Image"""
    if tensor_image.dim() == 4:
        tensor_image = tensor_image.squeeze(0)  # Remove batch dimension
    transform = T.ToPILImage()
    return transform(tensor_image.permute(2, 0, 1))

def get_image_caption(tensor_image, prompt="Describe this image in 10 words"):
    """Get caption for an image using Gemini Vision"""
    genai.configure(api_key='YOUR_GOOGLE_API_KEY')
    model = genai.GenerativeModel('gemini-pro-vision')
    
    pil_image = tensor_to_pil(tensor_image)
    response = model.generate_content([prompt, pil_image])
    
    return response.text

img = next(iter(dataloader))
caption = get_image_caption(img)
print(caption)    
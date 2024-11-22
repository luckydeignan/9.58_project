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
import os
from dotenv import load_dotenv
import google.generativeai as genai
from caption_prompts import get_caption_prompt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tenacity import retry, stop_after_attempt, wait_fixed
import time
from wordfreq import word_frequency
truncated = True

load_dotenv()

# Debug GPU availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))
    print("GPU Device Count:", torch.cuda.device_count())

# Remove or comment out GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')  # Force CPU usage
print(f"Using device: {device}")


# Class for Loading in numpy array of images and representing each as Torch Tensor
class ImageLoader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        # img.save('data/trained_images/image_{}.png'.format(idx)) used as sanity check that captions were working
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

# unnecessary code that was used when we had LLM analyze image
# def tensor_to_pil(tensor_image):
#     """Convert a PyTorch tensor to PIL Image"""
#     if tensor_image.dim() == 4:
#         tensor_image = tensor_image.squeeze(0)  # Remove batch dimension
#     transform = T.ToPILImage()
#     return transform(tensor_image.permute(2, 0, 1))


# function that creates image caption based on the image itself and prompt for LLM
# using Gemini 1.5 flash
# retry functionality in case calls per minute breaks quota
@retry(
    stop=stop_after_attempt(4),
    wait=wait_fixed(60)
)
def get_image_caption_with_retry(original_captions, prompt="Generate a caption with at least 10 words which summarizes the information given \
in the following descriptions:"):
    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-1.5-flash')
        for cap in original_captions:
            if cap:
                prompt = prompt + f"\ {cap}"
        
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        print(f"Retrying due to error: {e}")
        raise  # Re-raise the exception to trigger retry

# calls caption_generation function
# executes 24HR backoff if total requests per day API quota is reached
def attempt_with_24hr_backoff(original_captions):
    global truncated
    truncated = False
    while True:
        try:
            caption = get_image_caption_with_retry(original_captions)
            return caption  # If successful, return the result
        except Exception as e:
            print(f"All attempts failed. Waiting 24 hours before retrying. Error: {e}")
            time.sleep(86400)  # Wait for 24 hours before retrying again

# naive approach to caption generation -- taking 10 least-frequent words in caption
def truncated_captions(original_captions):
    global truncated
    truncated = True
    words = []
    for cap in original_captions:
        words = words + cap.split()
    for i, w in enumerate(words):
        if w[-1] == '.':
            words[i] = w[:-1]

    distinct_words = set(words)
    sorted_words = sorted(distinct_words, key= lambda word: word_frequency(word, 'en', wordlist='best'), reverse=False)
    final = ''
    for i in range(min(len(sorted_words), 10)):
        final = final + str(sorted_words[i]) + ' '

    return final

# load in caption numpy arrays (up to 5 descriptions per image)
train_caps = np.load('data/processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub))
# create captions for 50 first images of training data
captions = []
for idx, img in enumerate(dataloader):
    
    cap = truncated_captions(train_caps[idx])
    #toggle this to indicate modality of caption generation
    #cap = attempt_with_24hr_backoff(train_caps[idx])
    captions.append(cap)
    print(f'image {idx}: {cap}')

# create caption embeddings using general-purpose Hugging Face embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
model = model.to(device)  # Move model to GPU if available
embeddings = model.encode(captions, device=device)

# Apply PCA to reduce to 50 dimensions
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)

# Output shape will now be (num_sentences, 50)
print("Original shape:", embeddings.shape)          # (num_sentences, 384)
print("Reduced shape:", reduced_embeddings.shape)    # (num_sentences, 50)


directory = 'data/caption_embeddings/subj{:02d}'.format(sub)
os.makedirs(directory, exist_ok=True)

if truncated:
    np.save('{}/truncated_caption_bottleneck_embeddings_sub{}.npy'.format(directory,sub), reduced_embeddings)
else:
    np.save('{}/LLM_caption_bottleneck_embeddings_sub{}.npy'.format(directory,sub), reduced_embeddings)

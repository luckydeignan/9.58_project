import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON files
def load_metrics(caption_type):
    with open(f'results/evaluation_metrics_{caption_type}_captions/reconstruction_metrics_subj01.json', 'r') as f:
        return json.load(f)

# Load all metrics
llm_metrics = load_metrics('LLM')
short_metrics = load_metrics('short')
long_metrics = load_metrics('long')

# Extract mean values
caption_lengths = ['LLM', 'Short', 'Long', 'Wang et al.']
ssim_means = [
    llm_metrics['image_metrics']['ssim_mean'],
    short_metrics['image_metrics']['ssim_mean'],
    long_metrics['image_metrics']['ssim_mean'],
    0.23  # Wang et al. reference
]
pixcorr_means = [
    llm_metrics['image_metrics']['pixel_correlation_mean'],
    short_metrics['image_metrics']['pixel_correlation_mean'],
    long_metrics['image_metrics']['pixel_correlation_mean'],
    0.30  # Wang et al. reference
]
clip_means = [
    llm_metrics['image_metrics']['clip_score_mean'],
    short_metrics['image_metrics']['clip_score_mean'],
    long_metrics['image_metrics']['clip_score_mean'],
    0.64  # Wang et al. reference
]

# Set up the plot style
plt.style.use('seaborn')
width = 0.20  # Adjusted width to fit four bars
x = np.arange(len(caption_lengths))

# Create figure and axis with larger size
fig, ax = plt.subplots(figsize=(12, 6))

# Create bars
rects1 = ax.bar(x - width, ssim_means, width, label='SSIM', color='#4A90E2')
rects2 = ax.bar(x, pixcorr_means, width, label='Pixel Correlation', color='#50C878')
rects3 = ax.bar(x + width, clip_means, width, label='CLIP Score', color='#FFB6C1')

# Customize the plot
ax.set_ylabel('Score')
ax.set_title('Image Reconstruction Metrics by Caption Length')
ax.set_xticks(x)
ax.set_xticklabels(caption_lengths)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/reconstruction_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create box plots for distributions
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Prepare data for box plots
ssim_data = [
    llm_metrics['image_metrics']['ssim_all'],
    short_metrics['image_metrics']['ssim_all'],
    long_metrics['image_metrics']['ssim_all']
]

pixcorr_data = [
    llm_metrics['image_metrics']['pixel_correlation_all'],
    short_metrics['image_metrics']['pixel_correlation_all'],
    long_metrics['image_metrics']['pixel_correlation_all']
]

clip_data = [
    llm_metrics['image_metrics']['clip_scores_all'],
    short_metrics['image_metrics']['clip_scores_all'],
    long_metrics['image_metrics']['clip_scores_all']
]

# Create box plots
ax1.boxplot(ssim_data, labels=caption_lengths[:-1])
ax1.set_title('SSIM Distribution')
ax1.set_ylabel('SSIM Score')

ax2.boxplot(pixcorr_data, labels=caption_lengths[:-1])
ax2.set_title('Pixel Correlation Distribution')
ax2.set_ylabel('Correlation Score')

ax3.boxplot(clip_data, labels=caption_lengths[:-1])
ax3.set_title('CLIP Score Distribution')
ax3.set_ylabel('CLIP Score')

# Save the box plots
plt.tight_layout()
plt.savefig('results/reconstruction_metrics_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Create 2x2 comparison figure
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Load images
original = plt.imread('results/images/original.png')
llm = plt.imread('results/images/LLM.png')
short = plt.imread('results/images/short-naive.png')
long = plt.imread('results/images/long-naive.png')

# Plot images
axes[0, 0].imshow(original)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(short)
axes[0, 1].set_title('Short Caption')
axes[0, 1].axis('off')

axes[1, 0].imshow(long)
axes[1, 0].set_title('Long Caption')
axes[1, 0].axis('off')

axes[1, 1].imshow(llm)
axes[1, 1].set_title('LLM Caption')
axes[1, 1].axis('off')

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/image_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON files
def load_metrics(caption_type):
    with open(f'results/evaluation_metrics_{caption_type}_captions/reconstruction_metrics_subj01.json', 'r') as f:
        return json.load(f)

# Load all metrics
preliminary_metrics = load_metrics('preliminary')
short_metrics = load_metrics('short')
long_metrics = load_metrics('long')

# Extract mean values
caption_lengths = ['Preliminary', 'Short', 'Long']
ssim_means = [
    preliminary_metrics['image_metrics']['ssim_mean'],
    short_metrics['image_metrics']['ssim_mean'],
    long_metrics['image_metrics']['ssim_mean']
]
pixcorr_means = [
    preliminary_metrics['image_metrics']['pixel_correlation_mean'],
    short_metrics['image_metrics']['pixel_correlation_mean'],
    long_metrics['image_metrics']['pixel_correlation_mean']
]

# Set up the plot style
plt.style.use('seaborn')
width = 0.35
x = np.arange(len(caption_lengths))

# Create figure and axis with larger size
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
rects1 = ax.bar(x - width/2, ssim_means, width, label='SSIM', color='#4A90E2')
rects2 = ax.bar(x + width/2, pixcorr_means, width, label='Pixel Correlation', color='#50C878')

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

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/reconstruction_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create box plots for distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Prepare data for box plots
ssim_data = [
    preliminary_metrics['image_metrics']['ssim_all'],
    short_metrics['image_metrics']['ssim_all'],
    long_metrics['image_metrics']['ssim_all']
]

pixcorr_data = [
    preliminary_metrics['image_metrics']['pixel_correlation_all'],
    short_metrics['image_metrics']['pixel_correlation_all'],
    long_metrics['image_metrics']['pixel_correlation_all']
]

# Create box plots
ax1.boxplot(ssim_data, labels=caption_lengths)
ax1.set_title('SSIM Distribution')
ax1.set_ylabel('SSIM Score')

ax2.boxplot(pixcorr_data, labels=caption_lengths)
ax2.set_title('Pixel Correlation Distribution')
ax2.set_ylabel('Correlation Score')

# Save the box plots
plt.tight_layout()
plt.savefig('results/reconstruction_metrics_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

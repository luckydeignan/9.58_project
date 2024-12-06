import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-cap', '--cap_length', help='Caption length (short/long)', choices=["short", 'long', "LLM"], required=True)
args = parser.parse_args()
cap_length = args.cap_length

print("Loading ROI data")
roi_dir = f'data/processed_data/subj01/roi_{cap_length}_captions'
num_rois = 13
roi_act = np.zeros((num_rois, 15724)).astype(np.float32)

# Load ROIs
roi_act[0] = np.load(f"{roi_dir}/floc-faces.npy")
roi_act[1] = np.load(f"{roi_dir}/floc-words.npy")
roi_act[2] = np.load(f"{roi_dir}/floc-places.npy")
roi_act[3] = np.load(f"{roi_dir}/floc-bodies.npy")
roi_act[4] = np.load(f"{roi_dir}/V1.npy")
roi_act[5] = np.load(f"{roi_dir}/V2.npy")
roi_act[6] = np.load(f"{roi_dir}/V3.npy")
roi_act[7] = np.load(f"{roi_dir}/V4.npy")
roi_act[8] = np.load(f"{roi_dir}/ecc05.npy")
roi_act[9] = np.load(f"{roi_dir}/ecc10.npy")
roi_act[10] = np.load(f"{roi_dir}/ecc20.npy")
roi_act[11] = np.load(f"{roi_dir}/ecc40.npy")
roi_act[12] = np.load(f"{roi_dir}/ecc40p.npy")

# Binarize ROI masks
roi_act[roi_act>0]=1
roi_act[roi_act<0]=0

print("Loading caption embeddings")
if cap_length == 'LLM':
    train_captions = np.load('data/caption_embeddings/subj01/final_LLM_caption_bottleneck_embeddings_sub1.npy')
else:
    train_captions = np.load(f'data/caption_embeddings/subj01/{cap_length}er_truncated_caption_bottleneck_embeddings_sub1.npy')

# print('Training ROI to Caption Regression')
# print(f"ROI shape: {roi_act.shape}, Caption shape: {train_captions.shape}")
# print(f"First caption shape: {train_captions[0].shape}")
# print(f"ROI.T shape: {roi_act.T.shape}")

# Train regression from ROI space (15724) to caption space (50)
reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)

# Reshape ROI data to match caption dimensions
roi_expanded = np.repeat(roi_act, train_captions.shape[0] // roi_act.shape[0] + 1, axis=0)[:train_captions.shape[0]]
# print(f"Expanded ROI shape: {roi_expanded.shape}")  # Should be (8859, 15724)

reg.fit(roi_expanded, train_captions)  # Now dimensions match

# print(f"Regression coef shape: {reg.coef_.shape}")
# print(f"Regression intercept shape: {reg.intercept_.shape}")

datadict = {
    'weight': reg.coef_,  # Should be (50, 15724)
    'bias': reg.intercept_,
}

with open(f'data/regression_weights/subj01/{cap_length}_roi_to_caption_weights.pkl', "wb") as f:
    pickle.dump(datadict, f)

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time for ROI caption regression {cap_length}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
print('='*50)
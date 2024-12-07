import numpy as np
import lucas_kanade as lk
import cv2
from plot_utils import create_histogram, plot_mc_stats, plot_mc_error, plot_error
import os
from copy import copy
from tqdm import tqdm
import shutil
from utils import debug_messages

np.random.seed(11001)

input_video = "data/videos/blue_angels_formation.mp4"
start_frame = 150
output_dir = "./test_results/mc_analysis/blue_angels_formation/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Number of Monte Carlo runs
num_runs = 100

# Define distributions for each factor
distributions = {
    "max_corners":        lambda n: np.random.randint(2500, 5000, n),             # Uniform integer [500, 5000]
    "kernel_size":        lambda n: np.random.choice([3, 5, 7], n),     # Odd integers [3, 5, 7]
    "gaussian_sigma":     lambda n: np.round(np.random.uniform(0, 2, n), 6),     # Uniform float [0, 2]
    "corner_sensitivity": lambda n: np.round(np.random.uniform(0.005, 0.0005, n), 6), # Uniform float [0.005, 0.0001]
    "minimum_distance":   lambda n: np.random.randint(1, 6, n),                 # Uniform float [1, 5]
    "reinit_threshold":   lambda n: np.random.randint(10, 25, n),                 # Uniform float [0.01, 0.5]
    "window_size":        lambda n: np.random.choice([3, 5, 7], n),    # Odd integers [3, 5, 7]
    "error_threshold":    lambda n: np.round(np.random.uniform(0.7, 0.9, n), 6), # Uniform float [0.7, 0.9]
    "pyrdown_level":      lambda n: np.random.choice([2, 3, 4], n)                   # Uniform integer [2, 4]
}
headers = list(distributions.keys())
print(f"Parameters for Monte Carlo Analysis:\n{[i for i in headers]}\n")

mc_data = {factor: dist(num_runs) for factor, dist in distributions.items()}
create_histogram(mc_data, f"{output_dir}/mc_histogram.png")

mc_frames, mc_reinits, mc_reinits_cv2, mc_attempts, mc_attempts_cv2, mc_stc_corners, mc_stc_corners_cv2, mc_stc_maes, mc_stc_psnrs, mc_stc_ssims, mc_stc_precs, mc_stc_recs, mc_lk_good_corners, mc_lk_good_corners_cv2, mc_lk_maes, mc_lk_psnrs, mc_lk_ssims, mc_lk_precs, mc_lk_recs =  ([] for _ in range(19))
for i in tqdm(range(num_runs), desc="Processing Run"):
    
    if i == 0:
        save_samples = True
    else:
        save_samples = False
        
    output_path = os.path.join(output_dir, f"run_{i}")
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)    
    os.mkdir(output_path)
    
    mc_vars = []
    # print("\n")
    for key, vals in mc_data.items():
        # print(f"{key}: {vals[i]}")
        mc_vars.append(vals[i])
    
    lk_params = dict(winSize=(int(mc_vars[6]),int(mc_vars[6])), maxLevel=int(mc_vars[8]), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    feature_params = dict(max_corners=int(mc_vars[0]), ksize=int(mc_vars[1]), sensitivity=mc_vars[3], min_dist=int(mc_vars[4]), sigma0=mc_vars[2])
    
    output_video = os.path.join(output_path, input_video.split("/")[-1])

    frames, reinits, reinits_cv2, attempts, attempts_cv2, stc_corners, stc_corners_cv2, stc_maes, stc_psnrs, stc_ssims, stc_precs, stc_recs, lk_good_corners, lk_good_corners_cv2, lk_maes, lk_psnrs, lk_ssims, lk_precs, lk_recs = lk.lucas_kanade_optical_flow(input_video_path=input_video, output_video_path=output_video, lk_params=lk_params, feature_params=feature_params, reinit_threshold=int(mc_vars[5]), err_thresh=mc_vars[7], start_frame=start_frame, plots_dir=output_path, save_samples=True)
    
    mc_frames.append(frames)
    mc_reinits.append(reinits)
    mc_reinits_cv2.append(reinits_cv2)
    mc_attempts.append(attempts)
    mc_attempts_cv2.append(attempts_cv2)
    mc_stc_corners.append(stc_corners)
    mc_stc_corners_cv2.append(stc_corners_cv2)
    mc_stc_maes.append(stc_maes)
    mc_stc_psnrs.append(stc_psnrs)
    mc_stc_ssims.append(stc_ssims)
    mc_stc_precs.append(stc_precs)
    mc_stc_recs.append(stc_recs)
    mc_lk_good_corners.append(lk_good_corners)
    mc_lk_good_corners_cv2.append(lk_good_corners_cv2)
    mc_lk_maes.append(lk_maes)
    mc_lk_psnrs.append(lk_psnrs)
    mc_lk_ssims.append(lk_ssims)
    mc_lk_precs.append(lk_precs)
    mc_lk_recs.append(lk_recs)
    

output_file1 = f"{output_dir}/mc_reinits.png"
output_file2 = f"{output_dir}/mc_attempts.png"
output_file3 = f"{output_dir}/mc_stc_corners.png"
output_file4 = f"{output_dir}/mc_lk_good_corners.png"
output_file5 = f"{output_dir}/mc_stc_mae.png"
output_file6 = f"{output_dir}/mc_stc_psnr.png"
output_file7 = f"{output_dir}/mc_stc_ssim.png"
output_file8 = f"{output_dir}/mc_stc_precision.png"
output_file9 = f"{output_dir}/mc_stc_recall.png"        
output_file10 = f"{output_dir}/mc_lk_mae.png"
output_file11 = f"{output_dir}/mc_lk_psnr.png"
output_file12 = f"{output_dir}/mc_lk_ssim.png"
output_file13 = f"{output_dir}/mc_lk_precision.png"
output_file14 = f"{output_dir}/mc_lk_recall.png"

plot_mc_stats(mc_frames, mc_reinits, mc_reinits_cv2, "Detect New Corners", output_file=output_file1, title="Reinitialized Corners")
plot_mc_stats(mc_frames, mc_attempts, mc_attempts_cv2, "Attempted Optical Flow", output_file=output_file2, title="Attempts with Previous Frame's Corners")
plot_mc_stats(mc_frames, mc_stc_corners, mc_stc_corners_cv2, "Number of Detected Corners", output_file=output_file3, title="Shi Tomasi Corners")
plot_mc_stats(mc_frames, mc_lk_good_corners, mc_lk_good_corners_cv2, "Number of Good Corners", output_file=output_file4, title="Lucas Kanade Good Corners")

plot_mc_error(mc_frames, mc_stc_maes, "MAE", output_file=output_file5, title="Shi-Tomasi MAE")
plot_mc_error(mc_frames, mc_stc_psnrs, "PSNR", output_file=output_file6, title="Shi-Tomasi PSNR")
plot_mc_error(mc_frames, mc_stc_ssims, "SSIM", output_file=output_file7, title="Shi-Tomasi SSIM")
plot_mc_error(mc_frames, mc_stc_precs, "Precision", output_file=output_file8, title="Shi-Tomasi Precision")
plot_mc_error(mc_frames, mc_stc_recs, "Recall", output_file=output_file9, title="Shi-Tomasi Recall")
plot_mc_error(mc_frames, mc_lk_maes, "MAE", output_file=output_file10, title="Lucas Kanade MAE")
plot_mc_error(mc_frames, mc_lk_psnrs, "PSNR", output_file=output_file11, title="Lucas Kanade PSNR")
plot_mc_error(mc_frames, mc_lk_ssims, "SSIM", output_file=output_file12, title="Lucas Kanade SSIM")
plot_mc_error(mc_frames, mc_lk_precs, "Precision", output_file=output_file13, title="Lucas Kanade Precision")
plot_mc_error(mc_frames, mc_lk_recs, "Recall", output_file=output_file14, title="Lucas Kanade Recall")

for key, vals in mc_data.items():
    plot_error(np.arange(0, num_runs), vals, f"{key}", output_file=f"{output_dir}/mc_{key}.png", title=f"{key}")
    
    
#TODO
# Store average results in a dataframe and write to csv
# Determine the correlation between variables to see which is most sensitive
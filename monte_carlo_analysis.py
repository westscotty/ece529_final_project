import numpy as np
import lucas_kanade as lk
import cv2
from plot_utils import create_histogram
import os
from copy import copy
from tqdm import tqdm
import shutil
from utils import debug_messages

np.random.seed(11001)

input_video = "data/videos/blue_angels_formation.mp4"
start_frame = 190
output_dir = "./test_results/mc_analysis/blue_angels_formation/"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Number of Monte Carlo runs
num_runs = 1

# Define distributions for each factor
distributions = {
    "max_corners":        lambda n: np.random.randint(500, 5000, n),             # Uniform integer [500, 5000]
    "kernel_size":        lambda n: np.random.choice(np.arange(3, 8, 2), n),     # Odd integers [3, 5, 7]
    "gaussian_sigma":     lambda n: np.round(np.random.uniform(0, 3, n), 6),     # Uniform float [0, 3]
    "corner_sensitivity": lambda n: np.round(np.random.uniform(0.01, 0.0001, n), 6), # Uniform float [0.01, 0.0001]
    "minimum_distance":   lambda n: np.random.randint(1, 10, n),                 # Uniform float [1, 10]
    "reinit_threshold":   lambda n: np.random.randint(5, 25, n),                 # Uniform float [0.01, 0.5]
    "window_size":        lambda n: np.random.choice(np.arange(3, 10, 2), n),    # Odd integers [3, 5, 7, 9, 11]
    "error_threshold":    lambda n: np.round(np.random.uniform(0.7, 0.9, n), 6), # Uniform float [0.7, 0.9]
    "pyrdown_level":      lambda n: np.random.randint(2, 6, n)                   # Uniform integer [1, 5]
}
headers = list(distributions.keys())
print(f"Parameters for Monte Carlo Analysis:\n{[i for i in headers]}\n")

mc_data = {factor: dist(num_runs) for factor, dist in distributions.items()}
create_histogram(mc_data, f"{output_dir}/mc_histogram.png")

# monte_carlo_setup_data = np.column_stack([monte_carlo_raw_data[factor] for factor in distributions])

for i in tqdm(range(num_runs), desc="Processing Run"):
    
    if i == 0:
        save_samples = True
    else:
        save_samples = False
        
    output_path = os.path.join(output_dir, f"run_{i}")
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)    
    os.mkdir(output_path)
    
    # mc_vars = monte_carlo_setup_data[i]
    mc_vars = []
    print()
    for key, vals in mc_data.items():
        print(f"{key}: {vals[i]}")
        mc_vars.append(vals[i])
    
    lk_params = dict(winSize=(int(mc_vars[6]),int(mc_vars[6])), maxLevel=int(mc_vars[8]), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    feature_params = dict(max_corners=int(mc_vars[0]), ksize=int(mc_vars[1]), sensitivity=mc_vars[3], min_dist=int(mc_vars[4]), sigma0=mc_vars[2])
    
    output_video = os.path.join(output_path, input_video.split("/")[-1])

    lk.lucas_kanade_optical_flow(input_video_path=input_video, output_video_path=output_video, lk_params=lk_params, feature_params=feature_params, reinit_threshold=int(mc_vars[5]), err_thresh=mc_vars[7], start_frame=start_frame, plots_dir=output_path, save_samples=save_samples)
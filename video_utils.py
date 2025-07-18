import cv2
import numpy as np
from tqdm import tqdm
import utils as utils
import sys

green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
white = (255, 255, 255)

def open_video(video_path):
    
    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Error: Unable to open video file at {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return cap, frame_width, frame_height, total_frames

def get_crop_info(width, height):
    
    # Determine the size of the square crop based on the smaller dimension
    crop_size = min(width, height)
    
    # Calculate crop coordinates to center the crop within the larger dimension
    if width > height:
        crop_x_start = (width - crop_size) // 2
        crop_y_start = 0
    else:
        crop_x_start = 0
        crop_y_start = (height - crop_size) // 2
        
    return crop_size, crop_x_start, crop_y_start

def prepare_output_video(video_path, frame_rate, crop_size, pyrdown_level):
    
    # Setup VideoWriter for the cropped output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, frame_rate, (crop_size // 2**pyrdown_level, crop_size // 2**pyrdown_level), isColor=True)
    return out

def read_frame(cap, crop=False, crop_size=None, crop_x=None, crop_y=None, pyrdown_level=1, start_frame=1):
    
    for i in range(start_frame):
        ret, frame = cap.read()
    
    if not ret:
        sys.exit(f"Error: Unable to read the first frame.")
    
    if crop:
        frame = crop_frame(frame, crop_size, crop_x, crop_y)
     
    tmp_frame = frame.copy()
    if pyrdown_level > 0:
        for i in range(pyrdown_level):
            tmp_frame = cv2.pyrDown(tmp_frame)
        frame = tmp_frame

    return frame
    
def crop_frame(frame, crop_size, crop_x, crop_y):
    return frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

def reshape_points(points):
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2) if points.size > 0 else None
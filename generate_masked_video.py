import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Path Setup
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_dir, 'src'))

from data_loader import TrajectoryLoader
from masker import DynamicMasker

def generate_masked_video():
    print("==========================================")
    print("      GENERATING MASKED VIDEO DEMO        ")
    print("==========================================")

    # Load Data
    loader = TrajectoryLoader(dataroot='./data')
    scene = loader.get_valid_scene()
    
    # Setup Video Writer
    # Get first image to determine size
    samp = loader.nusc.get('sample', scene['first_sample_token'])
    cam_token = samp['data']['CAM_FRONT']
    cam_data = loader.nusc.get('sample_data', cam_token)
    first_img = cv2.imread(os.path.join(loader.dataroot, cam_data['filename']))
    h, w, _ = first_img.shape
    
    output_path = 'masked_features_demo.mp4'
    fps = 10 # Samples are 2Hz, but we can speed it up for viewing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Initialize Masker
    masker = DynamicMasker()
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Determine end time of the scene
    last_samp = loader.nusc.get('sample', scene['last_sample_token'])
    last_cam_token = last_samp['data']['CAM_FRONT']
    last_cam_data = loader.nusc.get('sample_data', last_cam_token)
    t_end_us = last_cam_data['timestamp']

    print(f"Processing Scene: {scene['name']} (All Frames)...")
    
    # Start from the first sample's camera data
    curr_token = samp['data']['CAM_FRONT']
    
    # We don't know the exact number of frames easily, so we'll just update the pbar
    pbar = tqdm()
    
    while curr_token != '':
        # Load Camera Data
        cam_data = loader.nusc.get('sample_data', curr_token)
        
        # Stop if we drift past the end of the scene
        if cam_data['timestamp'] > t_end_us + 100000: # 100ms buffer
            break

        img_path = os.path.join(loader.dataroot, cam_data['filename'])
        img = cv2.imread(img_path)
        
        if img is None:
            break
            
        # Iterate through framesk
        mask = masker.get_static_mask(img)
        
        # Detect Features (using the mask)
        kp = orb.detect(img, mask=mask)
        
        # 1. Red overlay for masked (dynamic) regions
        overlay = img.copy()
        overlay[mask == 0] = [0, 0, 255] # Red
        
        # Blend overlay
        alpha = 0.4
        vis = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        # 2. Draw Keypoints (Green)
        vis = cv2.drawKeypoints(vis, kp, None, color=(0, 255, 0), flags=0)
        
        # 3. Add Text
        cv2.putText(vis, f"Dynamic Object Filtering", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, f"Features: {len(kp)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write Frame
        out.write(vis)
        
        # Next
        curr_token = cam_data['next']
        pbar.update(1)
        
    pbar.close()
    out.release()
    print(f"\nDone! Video saved to: {output_path}")

if __name__ == "__main__":
    generate_masked_video()

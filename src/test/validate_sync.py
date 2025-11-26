import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Path Setup
current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_script_dir)
sys.path.append(src_dir)
project_root = os.path.dirname(src_dir)
DATAROOT = os.path.join(project_root, 'data')

from data_loader import TrajectoryLoader
from map_maker import MapMaker
from matcher import Matcher

def compute_reprojection_error(points_2d_detected, points_2d_projected):
    """Euclidean distance between matched points"""
    diff = points_2d_detected - points_2d_projected
    dist = np.linalg.norm(diff, axis=1)
    return np.mean(dist)

def validate_sync():
    print("=== VALIDATING SYNC (COST LANDSCAPE) ===")
    
    # 1. Setup
    loader = TrajectoryLoader(dataroot=DATAROOT)
    scene = loader.get_valid_scene()
    
    # 2. Build Map (Using CAM_FRONT)
    print("Building Map from Reference Camera...")
    # Using frames 0 and 4 ensures enough baseline for triangulation
    map_maker = MapMaker(loader, scene, ref_camera_name='CAM_FRONT')
    points_3d, _, _ = map_maker.build_local_map(0, 4)
    print(f"Map Built: {len(points_3d)} points.")
    
    # 3. Initialize Matcher (Using CAM_FRONT_LEFT)
    matcher = Matcher(loader, scene, target_camera='CAM_FRONT')
    
    # We will test against Frame 2 of the target camera
    target_frame_idx = 2
    img = matcher.get_target_image(target_frame_idx)
    h, w, _ = img.shape
    
    # --- GENERATE "GROUND TRUTH" MATCHES ---
    # Since we don't have perfect 2D labels in the target image, 
    # we will ASSUME the current sync (offset=0) is perfect for the sake of the test.
    # We project points with offset=0 and call those the "Detected Features".
    # Then we will drift the time and see error increase.
    
    print("Generating Ground Truth projections (Offset=0)...")
    gt_pixels, mask = matcher.project_points(points_3d, target_frame_idx, time_offset=0.0)
    
    # Filter only points that land on the image
    valid_indices = np.where(
        (mask) & (gt_pixels[:,0]>=0) & (gt_pixels[:,0]<w) & 
        (gt_pixels[:,1]>=0) & (gt_pixels[:,1]<h)
    )[0]
    
    gt_pixels = gt_pixels[valid_indices]
    points_3d_subset = points_3d[valid_indices]
    print(f"Valid Test Points: {len(gt_pixels)}")

    # 4. Sweep Time Offsets (The "Cost Landscape")
    print("\n--- Running Time Sweep (-100ms to +100ms) ---")
    offsets = np.linspace(-0.1, 0.1, 50) # 50 steps
    errors = []
    
    for dt in offsets:
        # Project using the shifted time
        proj_pixels, _ = matcher.project_points(points_3d_subset, target_frame_idx, time_offset=dt)
        
        # Compare against our "Ground Truth" (Offset=0)
        # In a real scenario, "gt_pixels" would come from an ORB feature matcher.
        # Here, we verify the GEOMETRY constraint.
        err = compute_reprojection_error(gt_pixels, proj_pixels)
        errors.append(err)
    
    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(offsets * 1000, errors, 'b-', linewidth=2)
    plt.scatter(offsets * 1000, errors, c='r', s=10)
    
    plt.title(f"Optimization Cost Landscape\nScene: {scene['name']}")
    plt.xlabel("Time Offset (ms)")
    plt.ylabel("Average Reprojection Error (pixels)")
    plt.grid(True)
    plt.axvline(x=0, color='g', linestyle='--', label='True Sync (0ms)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    validate_sync()
import sys
import os
import numpy as np

# Ensure Python can find the 'src' folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import TrajectoryLoader
from map_maker import MapMaker
from matcher import Matcher
from masker import DynamicMasker
from optimizer import TimeOptimizer
from visualizer import Visualizer

# === CONFIGURATION ===
DATAROOT = './data'  # Adjust if your data is deeper (e.g. data/nuscenes-mini)
REF_CAM = 'CAM_FRONT'         # Trusted Map Source
TARGET_CAM = 'CAM_FRONT_LEFT' # Camera to Synchronize
INJECTED_ERROR = 0.10        # We simulate that the target camera is 50ms LATE

def main():
    print("===TEMPORAL SYNCHRONIZER ===")
    
    # Load Data
    loader = TrajectoryLoader(dataroot=DATAROOT)
    scene = loader.get_valid_scene()
    
    # Build Map (Reference)
    print(f"\n[Step 1] Building Map from {REF_CAM}...")
    map_maker = MapMaker(loader, scene, ref_camera_name=REF_CAM)
    points_3d, _, _ = map_maker.build_local_map(0, 4)
    print(f" -> Map contains {len(points_3d)} landmarks.")

    # Setup Matcher (Target)
    print(f"\n[Step 2] Targeting {TARGET_CAM}...")
    matcher = Matcher(loader, scene, target_camera=TARGET_CAM)
    
    # Run "Corruption" Experiment
    # Simulate a time delay to test recovery capabilities.
    # The optimizer will attempt to find the offset that aligns the observation.
    
    print(f"\n[Step 3] SIMULATION: Injecting {INJECTED_ERROR*1000:.1f}ms error.")
    
    # --- DYNAMIC MASKING INTEGRATION ---
    print("[Step 3.5] Generating Dynamic Mask for Target Frame...")
    masker = DynamicMasker()
    target_frame_idx = 2
    target_img = matcher.get_target_image(target_frame_idx)
    dynamic_mask = masker.get_static_mask(target_img)
    # -----------------------------------

    # Init optimizer normally (Observation at offset=0.0)
    optimizer = TimeOptimizer(matcher, points_3d, target_frame_idx=target_frame_idx, dynamic_mask=dynamic_mask)
    
    if len(optimizer.observed_pixels) == 0:
        print("\nCRITICAL FAILURE: No overlap found between cameras.")
        print("Please check feature matching parameters in src/map_maker.py.")
        return

    # Shift the "Ground Truth Observation" by the injected error.
    # This simulates the camera taking the picture with a delay.
    corrupted_pixels, _ = matcher.project_points(
        optimizer.points_3d_subset, 
        optimizer.frame_idx, 
        time_offset=INJECTED_ERROR 
    )
    # Overwrite the optimizer's target
    optimizer.observed_pixels = corrupted_pixels
    
    # Run Optimization
    print("\n[Step 4] Running Solver...")
    estimated_offset = optimizer.run(search_window=(-0.2, 0.2))
    
    # Validation
    print("\n[Step 5] Results")
    print(f"True Artificial Delay: {INJECTED_ERROR*1000:.4f} ms")
    print(f"Recovered Offset:      {estimated_offset*1000:.4f} ms")
    
    error_residual = estimated_offset - INJECTED_ERROR
    print(f"Residual Error:        {error_residual*1000:.4f} ms")
    
    if abs(error_residual) < 0.001: # 1ms tolerance
        print("\nSUCCESS: Synchronization Recovered!")
    else:
        print("\nWARNING: Recovery drift is high.")

    # Generate Comparison Snapshots
    print("\n[Step 6] Generating Comparison Snapshots...")
    os.makedirs('results', exist_ok=True)
    vis = Visualizer(loader, scene, ref_cam=REF_CAM, target_cam=TARGET_CAM)
    
    # Random frame index for snapshot
    snapshot_idx = 15 
    
    # Snapshot 1: Desync (No Mask)
    vis.save_comparison_frame(
        offset_seconds=INJECTED_ERROR,
        use_masking=False,
        output_path='results/1_desync_nomask.jpg',
        frame_idx=snapshot_idx
    )
    
    # Snapshot 2: Desync (With Mask)
    vis.save_comparison_frame(
        offset_seconds=INJECTED_ERROR,
        use_masking=True,
        output_path='results/2_desync_masked.jpg',
        frame_idx=snapshot_idx
    )
    
    # Snapshot 3: Synchronized (With Mask)
    vis.save_comparison_frame(
        offset_seconds=INJECTED_ERROR - estimated_offset,
        use_masking=True,
        output_path='results/3_synchronized.jpg',
        frame_idx=snapshot_idx
    )

if __name__ == "__main__":
    main()
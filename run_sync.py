# FILE: run_sync.py
import sys
import os
import numpy as np

# Ensure Python can find the 'src' folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import TrajectoryLoader
from map_maker import MapMaker
from matcher import Matcher
from optimizer import TimeOptimizer

# === CONFIGURATION ===
DATAROOT = './data'  # Adjust if your data is deeper (e.g. data/nuscenes-mini)
REF_CAM = 'CAM_FRONT'         # Trusted Map Source
TARGET_CAM = 'CAM_FRONT_LEFT' # Camera to Synchronize
INJECTED_ERROR = 0.050        # We simulate that the target camera is 50ms LATE

def main():
    print("==========================================")
    print("   VISUAL SPATIO-TEMPORAL SYNCHRONIZER    ")
    print("==========================================")
    
    # 1. Load Data
    loader = TrajectoryLoader(dataroot=DATAROOT)
    scene = loader.get_valid_scene()
    
    # 2. Build Map (Reference)
    print(f"\n[Step 1] Building Map from {REF_CAM}...")
    map_maker = MapMaker(loader, scene, ref_camera_name=REF_CAM)
    points_3d, _, _ = map_maker.build_local_map(0, 4)
    print(f" -> Map contains {len(points_3d)} landmarks.")

    # 3. Setup Matcher (Target)
    print(f"\n[Step 2] Targeting {TARGET_CAM}...")
    matcher = Matcher(loader, scene, target_camera=TARGET_CAM)
    
    # 4. Run "Corruption" Experiment
    # We want to prove we can recover a 50ms error.
    # To do this, we modify the optimizer slightly:
    # We tell it: "The 'Observed' pixels are at T_true + 50ms."
    # Then we ask it: "Find the offset to align T_true to these pixels."
    
    print(f"\n[Step 3] SIMULATION: Injecting {INJECTED_ERROR*1000:.1f}ms error.")
    
    # Init optimizer normally (Observation at offset=0.0)
    optimizer = TimeOptimizer(matcher, points_3d, target_frame_idx=2)
    
    if len(optimizer.observed_pixels) == 0:
        print("\nCRITICAL FAILURE: No overlap found between cameras.")
        print("FIX: Open src/map_maker.py and increase matches[:200] to matches[:5000]")
        return

    # THE TRICK: We shift the "Ground Truth Observation" by the injected error.
    # This simulates the camera taking the picture 50ms late.
    corrupted_pixels, _ = matcher.project_points(
        optimizer.points_3d_subset, 
        optimizer.frame_idx, 
        time_offset=INJECTED_ERROR 
    )
    # Overwrite the optimizer's target
    optimizer.observed_pixels = corrupted_pixels
    
    # 5. Run Optimization
    print("\n[Step 4] Running Solver...")
    estimated_offset = optimizer.run(search_window=(-0.2, 0.2))
    
    # 6. Validation
    print("\n[Step 5] Results")
    print(f"True Artificial Delay: {INJECTED_ERROR*1000:.4f} ms")
    print(f"Recovered Offset:      {estimated_offset*1000:.4f} ms")
    
    error_residual = estimated_offset - INJECTED_ERROR
    print(f"Residual Error:        {error_residual*1000:.4f} ms")
    
    if abs(error_residual) < 0.001: # 1ms tolerance
        print("\nSUCCESS: Synchronization Recovered!")
    else:
        print("\nWARNING: Recovery drift is high.")

if __name__ == "__main__":
    main()
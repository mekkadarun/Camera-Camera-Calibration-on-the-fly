# FILE: src/test/test_data_loader.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- PATH SETUP ---
# 1. Get the directory where THIS script lives (src/test/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to find 'src' so we can import data_loader
src_dir = os.path.dirname(current_script_dir)
sys.path.append(src_dir)

# 3. Go up two levels to find 'CAMERACALIBAFR', then down into 'data'
project_root = os.path.dirname(src_dir)
DATAROOT = os.path.join(project_root, 'data')

# Now we can import because we added 'src' to system path
try:
    from data_loader import TrajectoryLoader
except ImportError:
    print("Error: Could not import data_loader.py. Make sure it exists in the 'src' folder.")
    sys.exit(1)

def test_loader():
    print("=== TESTING DATA LOADER ===")
    print(f"Looking for data in: {DATAROOT}")
    
    # 1. Initialize
    try:
        loader = TrajectoryLoader(dataroot=DATAROOT, version='v1.0-trainval')
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("Tip: Check if 'v1.0-trainval' folder exists inside your 'data' folder.")
        return

    # 2. Get Scene
    try:
        scene = loader.get_valid_scene()
        print(f"SUCCESS: Found scene '{scene['name']}'")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not find valid scene. {e}")
        return

    # 3. Test Extrinsics
    print("\n--- Testing Extrinsics (Body -> Camera) ---")
    try:
        # Note: nuScenes often uses names like 'CAM_FRONT'
        T_cam1 = loader.get_extrinsics(scene, 'CAM_FRONT')
        T_cam2 = loader.get_extrinsics(scene, 'CAM_FRONT_LEFT')
        
        diff = np.linalg.norm(T_cam1[:3, 3] - T_cam2[:3, 3])
        print(f"Distance between cameras: {diff:.3f} meters")
        
        if diff > 3.0 or diff < 0.1:
             print("WARNING: Distance seems suspicious. Check calibration.")
        else:
             print("Calibration looks reasonable.")
             
    except Exception as e:
        print(f"Error fetching extrinsics: {e}")

    # 4. Test Trajectory Interpolation
    print("\n--- Testing Trajectory Interpolation ---")
    try:
        get_pose, timestamps = loader.get_trajectory_interpolator(scene)
        
        if len(timestamps) == 0:
            print("Error: No timestamps found.")
            return

        t0 = timestamps[0]
        t1 = timestamps[min(10, len(timestamps)-1)] # Safe index
        
        pose0 = get_pose(t0)
        pose1 = get_pose(t1)
        
        move_dist = np.linalg.norm(pose1[:3, 3] - pose0[:3, 3])
        time_diff = t1 - t0
        
        print(f"Time Delta: {time_diff:.4f}s")
        print(f"Distance Moved: {move_dist:.4f} meters")
        
        if time_diff > 0:
            speed = move_dist / time_diff
            print(f"Calculated Speed: {speed:.2f} m/s")
    except Exception as e:
        print(f"Error in interpolation: {e}")
        return

    # 5. Visual Sanity Check
    print("\n--- Generating Trajectory Plot... ---")
    try:
        path_x = []
        path_y = []
        
        # Sample 100 points
        test_times = np.linspace(timestamps[0], timestamps[-1], 100)
        for t in test_times:
            p = get_pose(t)
            path_x.append(p[0, 3])
            path_y.append(p[1, 3])
            
        plt.figure(figsize=(8, 8))
        plt.plot(path_x, path_y, 'b-', label='Interpolated Path')
        plt.scatter(path_x[0], path_y[0], c='g', label='Start')
        plt.scatter(path_x[-1], path_y[-1], c='r', label='End')
        plt.title(f"Vehicle Path: {scene['name']}")
        plt.xlabel("X (Global)")
        plt.ylabel("Y (Global)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
        print("Plot generated successfully.")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    test_loader()
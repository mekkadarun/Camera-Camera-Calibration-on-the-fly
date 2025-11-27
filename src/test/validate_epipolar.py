import os
import sys
import cv2
import numpy as np

# Path Setup
current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_script_dir)
sys.path.append(src_dir)
project_root = os.path.dirname(src_dir)
DATAROOT = os.path.join(project_root, 'data')

from data_loader import TrajectoryLoader

def draw_lines(img1, img2, lines, pts1, pts2):
    """Draws epipolar lines on images."""
    r, c, _ = img1.shape
    
    # Convert to BGR if grayscale
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(map(int, pt1)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(map(int, pt2)), 5, color, -1)
        
    return img1, img2

def validate_epipolar():
    print("=== VALIDATING SPATIAL CALIBRATION (EPIPOLAR LINES) ===")
    
    # 1. Load Data
    loader = TrajectoryLoader(dataroot=DATAROOT)
    scene = loader.get_valid_scene()
    
    cam1_name = 'CAM_FRONT'
    cam2_name = 'CAM_FRONT_LEFT'
    
    print(f"Cameras: {cam1_name} -> {cam2_name}")
    
    # 2. Get Images (Frame 0)
    # We need images that are somewhat synced. Frame 0 of both is fine.
    samp = loader.nusc.get('sample', scene['first_sample_token'])
    
    data1 = loader.nusc.get('sample_data', samp['data'][cam1_name])
    data2 = loader.nusc.get('sample_data', samp['data'][cam2_name])
    
    img1 = cv2.imread(os.path.join(loader.dataroot, data1['filename']))
    img2 = cv2.imread(os.path.join(loader.dataroot, data2['filename']))
    
    # 3. Get Calibration
    # Intrinsics
    cs1 = loader.nusc.get('calibrated_sensor', data1['calibrated_sensor_token'])
    cs2 = loader.nusc.get('calibrated_sensor', data2['calibrated_sensor_token'])
    K1 = np.array(cs1['camera_intrinsic'])
    K2 = np.array(cs2['camera_intrinsic'])
    D1 = np.zeros(5) # nuScenes images are undistorted
    D2 = np.zeros(5)
    
    # Extrinsics (Cam -> Body)
    T_c1_b = loader.get_extrinsics(scene, cam1_name)
    T_c2_b = loader.get_extrinsics(scene, cam2_name)
    
    # Relative Transform: Cam1 -> Cam2
    # P_c2 = T_c2_b^-1 * T_c1_b * P_c1
    T_c1_c2 = np.linalg.inv(T_c2_b) @ T_c1_b
    
    R = T_c1_c2[:3, :3]
    T = T_c1_c2[:3, 3]
    
    print("Relative Rotation:\n", R)
    print("Relative Translation:\n", T)
    
    # 4. Stereo Rectification
    # This computes the transforms to make epipolar lines horizontal
    image_size = (img1.shape[1], img1.shape[0])
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0 # 0=Crop to valid, 1=Keep all
    )
    
    # 5. Remap Images
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    
    img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    # 6. Draw Horizontal Lines
    # If rectification is correct, features should align horizontally.
    # We'll just draw horizontal lines every 50 pixels.
    vis1 = img1_rect.copy()
    vis2 = img2_rect.copy()
    
    for y in range(0, image_size[1], 50):
        color = (0, 255, 0)
        cv2.line(vis1, (0, y), (image_size[0], y), color, 1)
        cv2.line(vis2, (0, y), (image_size[0], y), color, 1)
        
    # 7. Stack and Save
    # Resize for display if needed, but side-by-side is best
    comparison = np.hstack((vis1, vis2))
    
    # Draw a few specific features? 
    # Hard to do automatically without a matcher, but the lines are the main check.
    # If the user looks at the image, they should see objects (like the road horizon, or a car) 
    # crossing the green line at the same height in both images.
    
    output_path = os.path.join(project_root, 'epipolar_validation.jpg')
    cv2.imwrite(output_path, comparison)
    print(f"Saved validation image to: {output_path}")
    print("Check if features (e.g. horizon, lane markings) lie on the same horizontal green lines.")

if __name__ == "__main__":
    validate_epipolar()

import cv2
import numpy as np
import os

class Matcher:
    def __init__(self, loader, scene, target_camera='CAM_FRONT_LEFT'):
        self.loader = loader
        self.scene = scene
        self.target_cam = target_camera
        
        # Load Calibration for Target Camera
        self.K = self._get_intrinsics(self.target_cam)
        self.T_bc = loader.get_extrinsics(scene, self.target_cam) # Body -> Camera
        
        # Pre-calc Inverse (Cam -> Body)
        R_bc = self.T_bc[:3, :3]
        t_bc = self.T_bc[:3, 3]
        self.R_cb = R_bc.T
        self.t_cb = -self.R_cb @ t_bc

    def _get_intrinsics(self, camera_name):
        first_samp = self.loader.nusc.get('sample', self.scene['first_sample_token'])
        sd_token = first_samp['data'][camera_name]
        sd = self.loader.nusc.get('sample_data', sd_token)
        cs = self.loader.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        return np.array(cs['camera_intrinsic'])

    def project_points(self, points_3d, frame_index, time_offset=0.0):
        """
        Projects 3D world points into the Target Camera image.
        
        Args:
            points_3d: (N, 3) numpy array of world coordinates
            frame_index: Which frame of the target camera to use
            time_offset: The time shift to apply (in seconds).
                         If time_offset=0.05, we simulate the camera being 50ms late.
        
        Returns:
            projected_pixels: (N, 2) coordinates
            valid_mask: Boolean array of points that are actually in front of the camera
        """
        # 1. Get the Nominal Timestamp of the frame
        sample_token = self.scene['first_sample_token']
        samples = []
        curr = sample_token
        while curr != '':
            samples.append(curr)
            curr = self.loader.nusc.get('sample', curr)['next']
            if len(samples) > frame_index: break
            
        sample = self.loader.nusc.get('sample', samples[frame_index])
        cam_token = sample['data'][self.target_cam]
        cam_data = self.loader.nusc.get('sample_data', cam_token)
        
        nominal_time = cam_data['timestamp'] / 1e6
        
        # --- THE CORE LOGIC ---
        # We query the pose at (t + offset)
        # This effectively "slides" the camera along the trajectory
        query_time = nominal_time + time_offset
        
        get_pose, _ = self.loader.get_trajectory_interpolator(self.scene)
        T_wb = get_pose(query_time) # World -> Body at shifted time
        
        # 2. Compute World -> Camera Transform
        # T_wc = T_wb * T_bc
        T_wc = T_wb @ self.T_bc
        
        # We need T_cw (World -> Camera) to project points
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        
        # 3. Project Points
        # P_cam = R_cw * P_world + t_cw
        points_cam = (R_cw @ points_3d.T).T + t_cw
        
        # Filter points behind the camera (Z < 0)
        valid_mask = points_cam[:, 2] > 0.1
        
        # Apply Intrinsic Matrix (K)
        # u = fx * X/Z + cx
        # v = fy * Y/Z + cy
        uv = (self.K @ points_cam.T).T
        uv[:, 0] /= uv[:, 2] # Divide by Z
        uv[:, 1] /= uv[:, 2]
        
        return uv[:, :2], valid_mask

    def get_target_image(self, frame_index):
        """Helper to load the image for visualization"""
        sample_token = self.scene['first_sample_token']
        samples = []
        curr = sample_token
        while curr != '':
            samples.append(curr)
            curr = self.loader.nusc.get('sample', curr)['next']
            if len(samples) > frame_index: break
            
        sample = self.loader.nusc.get('sample', samples[frame_index])
        cam_token = sample['data'][self.target_cam]
        cam_data = self.loader.nusc.get('sample_data', cam_token)
        
        path = os.path.join(self.loader.dataroot, cam_data['filename'])
        return cv2.imread(path)
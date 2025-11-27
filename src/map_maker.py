import cv2
import numpy as np
import os
from masker import DynamicMasker

class MapMaker:
    def __init__(self, loader, scene, ref_camera_name='CAM_FRONT_LEFT'):
        self.loader = loader
        self.scene = scene
        self.nusc = loader.nusc
        self.ref_camera = ref_camera_name
        
        print(f"[MapMaker] Initializing with Reference Camera: {self.ref_camera}")
        
        # Load Intrinsics (K) and Extrinsics (T_bc) for the chosen camera
        self.K = self._get_intrinsics(self.ref_camera)
        self.T_bc = loader.get_extrinsics(scene, self.ref_camera) 
        
        # Pre-calculate Camera->Body (Inverse extrinsics)
        R_bc = self.T_bc[:3, :3]
        t_bc = self.T_bc[:3, 3]
        self.R_cb = R_bc.T
        self.t_cb = -self.R_cb @ t_bc
        
        # Initialize Masker
        self.masker = DynamicMasker()

    def _get_intrinsics(self, camera_name):
        """Fetch Camera Calibration Matrix (K)."""
        first_samp = self.nusc.get('sample', self.scene['first_sample_token'])
        sd_token = first_samp['data'][camera_name]
        sd = self.nusc.get('sample_data', sd_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        return np.array(cs['camera_intrinsic'])

    def get_image_and_pose(self, sample_token):
        """Helper to load image and its exact pose."""
        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data'][self.ref_camera]
        cam_data = self.nusc.get('sample_data', cam_token)
        
        # Load Image
        filename = cam_data['filename']
        path = os.path.join(self.loader.dataroot, filename)
        
        if not os.path.exists(path):
            print(f"[DEBUG] Image missing at: {path}")
            
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {path}")

        # Get Pose
        timestamp = cam_data['timestamp'] / 1e6
        get_pose, _ = self.loader.get_trajectory_interpolator(self.scene)
        T_wb = get_pose(timestamp) 

        return img, T_wb

    def build_local_map(self, frame_idx_1=0, frame_idx_2=4):
        """Creates 3D points using two frames."""
        # Get Tokens
        sample_token = self.scene['first_sample_token']
        samples = []
        curr = sample_token
        while curr != '':
            samples.append(curr)
            curr = self.nusc.get('sample', curr)['next']
            if len(samples) > frame_idx_2: break
        
        # Load Data
        print(f"[MapMaker] Loading Frame {frame_idx_1} and Frame {frame_idx_2}...")
        img1, T_wb1 = self.get_image_and_pose(samples[frame_idx_1])
        img2, T_wb2 = self.get_image_and_pose(samples[frame_idx_2])
        
        # Generate Masks (Dynamic Object Filtering)
        print("[MapMaker] Generating Semantic Masks...")
        mask1 = self.masker.get_static_mask(img1)
        mask2 = self.masker.get_static_mask(img2)

        # Detect Features (ORB) with Mask
        orb = cv2.ORB_create(nfeatures=20000, fastThreshold=0)
        kp1, des1 = orb.detectAndCompute(img1, mask=mask1)
        kp2, des2 = orb.detectAndCompute(img2, mask=mask2)

        # Match Features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:10000]
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Projection Matrices
        P1 = self._get_projection_matrix(T_wb1)
        P2 = self._get_projection_matrix(T_wb2)

        # Triangulate
        pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        pts3d = pts4d[:3] / pts4d[3] 

        return pts3d.T, img1, pts1

    def _get_projection_matrix(self, T_wb):
        """Compute P = K * inv(T_wb * T_bc)"""
        T_wc = T_wb @ self.T_bc
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        RT = np.hstack((R_cw, t_cw.reshape(3, 1)))
        return self.K @ RT
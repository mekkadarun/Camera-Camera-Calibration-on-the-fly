import os
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation as R

class TrajectoryLoader:
    def __init__(self, dataroot, version='v1.0-trainval'):
        if not os.path.exists(dataroot):
            raise FileNotFoundError(f"Data root not found at: {dataroot}")
        
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot

    def get_valid_scene(self):
        """Scans for a scene where BOTH camera images exist on disk."""
        print("[TrajectoryLoader] Searching for a scene with CAM_FRONT and CAM_FRONT_LEFT...")
        for scene in self.nusc.scene:
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            
            # Check CAM_FRONT
            token_front = first_sample['data']['CAM_FRONT']
            data_front = self.nusc.get('sample_data', token_front)
            path_front = os.path.join(self.dataroot, data_front['filename'])

            # Check CAM_FRONT_LEFT
            token_side = first_sample['data']['CAM_FRONT_LEFT']
            data_side = self.nusc.get('sample_data', token_side)
            path_side = os.path.join(self.dataroot, data_side['filename'])
            
            # Only return if BOTH exist
            if os.path.exists(path_front) and os.path.exists(path_side):
                print(f"[TrajectoryLoader] Found valid scene: {scene['name']}")
                return scene
        
        raise FileNotFoundError("Could not find a scene where both CAM_FRONT and CAM_FRONT_LEFT images exist.")

    def get_extrinsics(self, scene, camera_name='CAM_FRONT_LEFT'):
        """Returns the 4x4 T_body_camera matrix."""
        first_samp = self.nusc.get('sample', scene['first_sample_token'])
        sd_token = first_samp['data'][camera_name]
        sd = self.nusc.get('sample_data', sd_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        
        T_bc = np.eye(4)
        # nuScenes: [w, x, y, z] -> SciPy: [x, y, z, w]
        q = cs['rotation']
        r_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T_bc[:3, :3] = r_mat
        T_bc[:3, 3] = cs['translation']
        
        return T_bc

    def get_trajectory_interpolator(self, scene):
        """Returns a function get_pose(t) based ONLY on Camera timestamps."""
        # 1. Determine time boundaries
        first_sample = self.nusc.get('sample', scene['first_sample_token'])
        last_sample = self.nusc.get('sample', scene['last_sample_token'])
        
        # We use CAM_FRONT as the reference timeline
        t_start_us = self.nusc.get('sample_data', first_sample['data']['CAM_FRONT'])['timestamp']
        t_end_us = self.nusc.get('sample_data', last_sample['data']['CAM_FRONT'])['timestamp']
        
        # 2. Collect ego_poses traversing the CAM_FRONT linked list
        timestamps = []
        positions = []
        quaternions = [] 

        # Start from CAM_FRONT instead of LIDAR_TOP
        curr_token = first_sample['data']['CAM_FRONT']
        
        while curr_token != '':
            record = self.nusc.get('sample_data', curr_token)
            
            # Stop if we drift past the end of the scene
            if record['timestamp'] > t_end_us + 100000: 
                break

            pose = self.nusc.get('ego_pose', record['ego_pose_token'])
            
            timestamps.append(pose['timestamp'] / 1e6)
            positions.append(pose['translation'])
            quaternions.append(pose['rotation'])
            
            curr_token = record['next']

        times = np.array(timestamps)
        pos = np.array(positions)
        quats = np.array(quaternions)

        # 3. Setup Interpolators
        pos_interp = interp1d(times, pos, axis=0, kind='linear', fill_value="extrapolate")
        
        quats_scipy = quats[:, [1, 2, 3, 0]] 
        rot_interp = Slerp(times, R.from_quat(quats_scipy))

        t_start = times[0]
        t_end = times[-1]

        def get_pose_at_time(t):
            t_clamped = np.clip(t, t_start, t_end)
            
            p = pos_interp(t) 
            r = rot_interp(t_clamped).as_matrix() 
            
            T = np.eye(4)
            T[:3, :3] = r
            T[:3, 3] = p
            return T

        return get_pose_at_time, times
import cv2
import numpy as np
import os
from tqdm import tqdm
from masker import DynamicMasker

class Visualizer:
    def __init__(self, loader, scene, ref_cam='CAM_FRONT', target_cam='CAM_FRONT_LEFT'):
        self.loader = loader
        self.scene = scene
        self.ref_cam = ref_cam
        self.target_cam = target_cam
        self.masker = DynamicMasker()
        self.orb = cv2.ORB_create(nfeatures=1000)

    def generate_comparison_video(self, offset_seconds, use_masking, output_path):
        """
        Generates a side-by-side video of Ref vs Target camera.
        
        Args:
            offset_seconds: Time shift to apply to Target Camera (simulates desync or sync).
            use_masking: If True, masks out dynamic objects before feature detection.
            output_path: Path to save the video.
        """
        print(f"[Visualizer] Generating video: {output_path}")
        print(f" -> Offset: {offset_seconds*1000:.1f}ms | Masking: {use_masking}")

        # Setup Video Writer
        # Get first image to determine size
        first_samp = self.loader.nusc.get('sample', self.scene['first_sample_token'])
        
        # Ref Cam Data
        ref_token = first_samp['data'][self.ref_cam]
        ref_data = self.loader.nusc.get('sample_data', ref_token)
        ref_img = cv2.imread(os.path.join(self.loader.dataroot, ref_data['filename']))
        
        h, w, _ = ref_img.shape
        # Side-by-side width
        out_w = w * 2
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, h))
        
        # Iterate through Reference Camera Frames
        curr_token = ref_token
        
        # Determine end time for progress bar
        last_samp = self.loader.nusc.get('sample', self.scene['last_sample_token'])
        t_end = self.loader.nusc.get('sample_data', last_samp['data'][self.ref_cam])['timestamp']
        
        pbar = tqdm()
        
        while curr_token != '':
            # Load Reference Frame
            ref_data = self.loader.nusc.get('sample_data', curr_token)
            if ref_data['timestamp'] > t_end + 100000: break
            
            ref_path = os.path.join(self.loader.dataroot, ref_data['filename'])
            img_ref = cv2.imread(ref_path)
            
            if img_ref is None: break
            
            # Process Frame
            combined = self._process_frame(img_ref, ref_data['timestamp'], offset_seconds, use_masking)
            out.write(combined)
            
            # Next
            curr_token = ref_data['next']
            pbar.update(1)
            
        pbar.close()
        out.release()
        print("Video saved.")

    def save_comparison_frame(self, offset_seconds, use_masking, output_path, frame_idx=10):
        """
        Saves a single side-by-side comparison image.
        """
        print(f"[Visualizer] Saving snapshot: {output_path}")
        
        # Get the N-th frame from the reference camera
        sample_token = self.scene['first_sample_token']
        samples = []
        curr = sample_token
        while curr != '':
            samples.append(curr)
            curr = self.loader.nusc.get('sample', curr)['next']
            if len(samples) > frame_idx: break
            
        # We want the sample_data (camera frame) corresponding to this sample
        # Or just iterate sample_data 'frame_idx' times?
        # Let's just grab the sample's camera token.
        samp = self.loader.nusc.get('sample', samples[-1])
        ref_token = samp['data'][self.ref_cam]
        ref_data = self.loader.nusc.get('sample_data', ref_token)
        
        ref_path = os.path.join(self.loader.dataroot, ref_data['filename'])
        img_ref = cv2.imread(ref_path)
        
        combined = self._process_frame(img_ref, ref_data['timestamp'], offset_seconds, use_masking)
        cv2.imwrite(output_path, combined)

    def _process_frame(self, img_ref, ref_timestamp, offset_seconds, use_masking):
        # B. Find Corresponding Target Frame (with Offset)
        target_ts = ref_timestamp + (offset_seconds * 1e6)
        img_target = self._get_closest_image(self.target_cam, target_ts)
        
        if img_target is None:
            img_target = np.zeros_like(img_ref)

            # Process Images (Masking & Features)
        if use_masking:
            mask_ref = self.masker.get_static_mask(img_ref)
            mask_target = self.masker.get_static_mask(img_target)
            
            # Visualize Mask (Red Overlay)
            overlay_ref = img_ref.copy()
            overlay_ref[mask_ref == 0] = [0, 0, 255]
            img_ref = cv2.addWeighted(img_ref, 0.7, overlay_ref, 0.3, 0)
            
            overlay_target = img_target.copy()
            overlay_target[mask_target == 0] = [0, 0, 255]
            img_target = cv2.addWeighted(img_target, 0.7, overlay_target, 0.3, 0)
            
            # Detect Features with Mask
            kp_ref = self.orb.detect(img_ref, mask=mask_ref)
            kp_target = self.orb.detect(img_target, mask=mask_target)
        else:
            # Detect Features without Mask
            kp_ref = self.orb.detect(img_ref, None)
            kp_target = self.orb.detect(img_target, None)
            
        # Draw Keypoints
        img_ref = cv2.drawKeypoints(img_ref, kp_ref, None, color=(0, 255, 0), flags=0)
        img_target = cv2.drawKeypoints(img_target, kp_target, None, color=(0, 255, 0), flags=0)
                # Combine and Write
            # Add Labels
        cv2.putText(img_ref, f"{self.ref_cam}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_target, f"{self.target_cam}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if offset_seconds != 0:
                cv2.putText(img_target, f"Offset: {offset_seconds*1000:.0f}ms", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return np.hstack((img_ref, img_target))

    def _get_closest_image(self, camera_name, target_ts_us):
        """
        Finds the image in 'camera_name' with timestamp closest to 'target_ts_us'.
        Optimized to search locally if we track state, but for now we'll do a 
        simple search starting from the beginning of the scene (slow but safe).
        
        Actually, to be faster:
        We can collect all timestamps for this camera in this scene once.
        """
        if not hasattr(self, '_cam_timestamps'):
            self._cam_timestamps = {}
            self._cam_tokens = {}
            
        if camera_name not in self._cam_timestamps:
            # Index this camera
            timestamps = []
            tokens = []
            
            first_samp = self.loader.nusc.get('sample', self.scene['first_sample_token'])
            curr = first_samp['data'][camera_name]
            
            # We need to go back to the very start of the scene data, 
            # not just the first sample, to be safe? 
            # Actually, first_sample is usually the start.
            
            while curr != '':
                sd = self.loader.nusc.get('sample_data', curr)
                timestamps.append(sd['timestamp'])
                tokens.append(curr)
                curr = sd['next']
                
            self._cam_timestamps[camera_name] = np.array(timestamps)
            self._cam_tokens[camera_name] = np.array(tokens)
            
        # Find closest
        times = self._cam_timestamps[camera_name]
        idx = (np.abs(times - target_ts_us)).argmin()
        
        # Check if it's reasonably close (e.g. within 100ms)
        # If it's too far, maybe we ran out of data?
        if np.abs(times[idx] - target_ts_us) > 200000: # 200ms
            return None
            
        best_token = self._cam_tokens[camera_name][idx]
        sd = self.loader.nusc.get('sample_data', best_token)
        path = os.path.join(self.loader.dataroot, sd['filename'])
        return cv2.imread(path)

import numpy as np
from scipy.optimize import minimize_scalar

class TimeOptimizer:
    def __init__(self, matcher, map_points, target_frame_idx, dynamic_mask=None):
        self.matcher = matcher
        self.map_points = map_points
        self.frame_idx = target_frame_idx
        
        # Establish the "Ground Truth" observation
        # In a real scenario, this comes from the image features.
        # For this prototype, we use the raw dataset (which is synced) as our anchor.
        # We will assume the raw data has offset = 0.0.
        self.observed_pixels, self.mask = matcher.project_points(
            map_points, target_frame_idx, time_offset=0.0
        )
        
        # Filter valid points (must be inside image bounds)
        h, w, _ = matcher.get_target_image(target_frame_idx).shape
        
        # Basic validity: Inside image + In front of camera
        valid = (self.mask) & \
                (self.observed_pixels[:,0]>=0) & (self.observed_pixels[:,0]<w) & \
                (self.observed_pixels[:,1]>=0) & (self.observed_pixels[:,1]<h)

        # Dynamic Object Filtering (Optional)
        if dynamic_mask is not None:
            # Check if points fall on dynamic objects (mask == 0)
            # We need integer coordinates to index the mask
            u = np.clip(self.observed_pixels[:, 0].astype(int), 0, w-1)
            v = np.clip(self.observed_pixels[:, 1].astype(int), 0, h-1)
            
            # Mask value at these pixels
            mask_vals = dynamic_mask[v, u]
            
            # Keep only if mask is 255 (Static)
            valid = valid & (mask_vals == 255)
            print(f"[Optimizer] Applied Dynamic Masking. {np.sum(mask_vals == 0)} points removed.")
        
        self.valid_indices = np.where(valid)[0]
        self.observed_pixels = self.observed_pixels[self.valid_indices]
        self.points_3d_subset = map_points[self.valid_indices]
        
        print(f"[Optimizer] Optimizing over {len(self.observed_pixels)} valid correspondences.")

    def objective_function(self, dt):
        """
        Cost = Average Pixel Distance between Observation and Prediction(dt)
        """
        # Project using the guess 'dt'
        proj_pixels, _ = self.matcher.project_points(
            self.points_3d_subset, self.frame_idx, time_offset=dt
        )
        
        # Calculate Error
        diff = self.observed_pixels - proj_pixels
        dist = np.linalg.norm(diff, axis=1)
        return np.mean(dist)

    def run(self, search_window=(-0.2, 0.2)):
        """Minimizes the objective function to find the time offset."""
        print(f"[Optimizer] Searching range {search_window}s...")
        
        # Bounded Search (Golden Section)
        res = minimize_scalar(
            self.objective_function, 
            bounds=search_window, 
            method='bounded',
            options={'xatol': 1e-5} # Precision: 0.01ms
        )
        
        if res.success:
            return res.x
        else:
            print("[Optimizer] Failed to converge.")
            return None
# FILE: src/optimizer.py
import numpy as np
from scipy.optimize import minimize_scalar

class TimeOptimizer:
    def __init__(self, matcher, map_points, target_frame_idx):
        self.matcher = matcher
        self.map_points = map_points
        self.frame_idx = target_frame_idx
        
        # 1. Establish the "Ground Truth" observation
        # In a real scenario, this comes from the image features.
        # For this prototype, we use the raw dataset (which is synced) as our anchor.
        # We will assume the raw data has offset = 0.0.
        self.observed_pixels, self.mask = matcher.project_points(
            map_points, target_frame_idx, time_offset=0.0
        )
        
        # 2. Filter valid points (must be inside image bounds)
        h, w, _ = matcher.get_target_image(target_frame_idx).shape
        valid = (self.mask) & \
                (self.observed_pixels[:,0]>=0) & (self.observed_pixels[:,0]<w) & \
                (self.observed_pixels[:,1]>=0) & (self.observed_pixels[:,1]<h)
        
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
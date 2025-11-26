import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Path Setup (Same as before)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_script_dir)
sys.path.append(src_dir)
project_root = os.path.dirname(src_dir)
DATAROOT = os.path.join(project_root, 'data')

from data_loader import TrajectoryLoader
from map_maker import MapMaker

def test_map():
    print("=== TESTING MAP MAKER ===")
    loader = TrajectoryLoader(dataroot=DATAROOT)
    scene = loader.get_valid_scene()
    
    mapper = MapMaker(loader, scene)
    
    # Try to triangulate between Frame 0 and Frame 6 (~0.5s gap)
    print("Triangulating points between Frame 0 and Frame 6...")
    try:
        points_3d, img, pixels = mapper.build_local_map(0, 6)
        
        print(f"Success! Generated {len(points_3d)} 3D landmarks.")
        
        # Filter Garbage (Behind camera or too far)
        # In nuScenes, Z is Up. Camera looks X-forward. 
        # Actually, in Camera Frame: Z is forward.
        # But our points are in WORLD frame.
        
        # Simple visualization
        fig = plt.figure(figsize=(10,5))
        
        # Plot 1: The Image with features
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.scatter(pixels[:,0,0], pixels[:,0,1], c='r', s=2)
        ax1.set_title("Tracked Features (Cam 1)")
        
        # Plot 2: The 3D Cloud (Top Down View)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(points_3d[:,0], points_3d[:,1], s=1)
        ax2.set_title("Triangulated Map (World X-Y)")
        ax2.set_xlabel("X (Global)")
        ax2.set_ylabel("Y (Global)")
        ax2.axis('equal')
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_map()
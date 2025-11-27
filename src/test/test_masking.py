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

from masker import DynamicMasker

def test_masking():
    print("=== TESTING DYNAMIC OBJECT MASKING ===")
    
    # 1. Load Data
    img_path = os.path.join(project_root, 'data/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151076362404.jpg')
    
    print(f"Loading Image: {img_path}")
    img = cv2.imread(img_path)
    
    # 2. Run Masker
    masker = DynamicMasker()
    print("Running DeepLabV3 Inference...")
    mask = masker.get_static_mask(img)
    
    # 3. Visualize
    # Mask is 255 (Static) and 0 (Dynamic).
    # Let's create an overlay: Red where masked out.
    
    overlay = img.copy()
    # Where mask is 0 (Dynamic), make it Red
    overlay[mask == 0] = [0, 0, 255] 
    
    # Blend
    alpha = 0.5
    vis = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    output_path = os.path.join(project_root, 'test_masking_output.jpg')
    cv2.imwrite(output_path, vis)
    
    print(f"Saved visualization to: {output_path}")
    print("Check the image: Cars/People should be highlighted in RED.")

if __name__ == "__main__":
    test_masking()

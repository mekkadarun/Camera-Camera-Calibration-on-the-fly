import torch
import torchvision
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image

class DynamicMasker:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"[DynamicMasker] Loading DeepLabV3 on {self.device}...")
        
        # Load Pre-trained Model
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        ).to(self.device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # COCO/Pascal VOC classes to mask out (Dynamic Objects)
        self.dynamic_classes = [
            2,  # bicycle
            6,  # bus
            7,  # car
            14, # motorbike
            15, # person
            19  # train
        ]

    def get_static_mask(self, image_bgr):
        """
        Returns a binary mask where 255 = Static (Keep), 0 = Dynamic (Ignore).
        """
        # Convert BGR -> RGB -> PIL
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get prediction (H, W)
        pred = output.argmax(0).byte().cpu().numpy()
        
        # Create Mask: 1 if class is NOT dynamic, 0 if dynamic
        mask = np.ones_like(pred, dtype=np.uint8) * 255
        
        for cls_idx in self.dynamic_classes:
            mask[pred == cls_idx] = 0
            
        return mask

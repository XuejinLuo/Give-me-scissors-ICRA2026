# mask_generator.py
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2  
import matplotlib.pyplot as plt
from functions.utils import show_anns
import os

class MaskGenerator(object):
    def __init__(self, config, roi_image_path):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sam = sam_model_registry["vit_b"](checkpoint="/home/luo/ICRA/models/sam_vit_b_01ec64.pth")
        self.sam.to(device=self.device)
        self.roi_image_path = roi_image_path

    def generate_masks(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask_generator = SamAutomaticMaskGenerator(sam)
        # masks = mask_generator.generate(image)

        mask_generator_2 = SamAutomaticMaskGenerator(
        model=self.sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        )
        masks = mask_generator_2.generate(image)
        return masks

    def _Generate_masks(self, save_path):
        ### Generate masks
        print("Generating masks...")
        masks = self.generate_masks(self.roi_image_path)
        print(f"Number of masks generated: {len(masks)}")
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.imread(self.roi_image_path))
        show_anns(masks)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, "image_masks.png"))
        plt.show()

        # original_image = cv2.imread(self.roi_image_path)
        # for mask in masks:
        #     mask_array = mask['segmentation']
        #     mask_image = np.zeros_like(original_image)
        #     mask_image[mask_array] = [255, 0, 0]
        #     overlay = cv2.addWeighted(original_image, 0.5, mask_image, 0.5, 0)
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(overlay)
        #     plt.axis('off')
        #     plt.show()

        return masks

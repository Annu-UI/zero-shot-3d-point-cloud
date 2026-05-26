import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

class ObjectSegmentor:
    def __init__(self, checkpoint_path="models/sam_vit_h_4b8939.pth"):
        """
        Initialize SAM (Segment Anything Model).
        
        SAM takes a bounding box as a prompt and returns a pixel-level mask.
        This is more precise than just using the bounding box directly.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found at {checkpoint_path}\n"
                f"Download it by running:\n"
                f"  from src.segmentation import download_sam\n"
                f"  download_sam()"
            )
        
        # Detect device — use GPU if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Segmentor] Using device: {self.device}")
        
        # Load SAM model — vit_h is the largest and most accurate
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(self.device)
        
        self.predictor = SamPredictor(sam)
        print("[Segmentor] SAM loaded successfully")

    def segment(self, image, box):
        """
        Generate pixel mask for object inside bounding box.
        
        Args:
            image: RGB numpy array (H, W, 3)
            box: [x1, y1, x2, y2] bounding box from YOLO-World
            
        Returns:
            mask: boolean numpy array (H, W) — True where object is
            masked_image: image with background removed
        """
        # SAM needs the image set before prediction
        self.predictor.set_image(image)
        
        # Convert box to numpy array SAM expects
        box_array = np.array(box)
        
        # Predict mask
        # SAM returns 3 mask candidates — we take the best one (index 0)
        masks, scores, _ = self.predictor.predict(
            box=box_array,
            multimask_output=True  # get 3 options, pick best
        )
        
        # Pick mask with highest confidence score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        print(f"[Segmentor] Mask generated — {mask.sum()} pixels selected "
              f"(confidence: {scores[best_idx]:.3f})")
        
        # Apply mask to image — set background to black
        masked_image = image.copy()
        masked_image[~mask] = 0
        
        return mask, masked_image

    def segment_all_boxes(self, image, boxes, labels):
        """
        Segment multiple detected objects.
        
        Args:
            image: RGB numpy array
            boxes: list of [x1, y1, x2, y2]
            labels: list of class names
            
        Returns:
            combined_mask: union of all object masks
            results: list of (label, mask, masked_image) per object
        """
        self.predictor.set_image(image)
        
        H, W = image.shape[:2]
        combined_mask = np.zeros((H, W), dtype=bool)
        results = []
        
        for box, label in zip(boxes, labels):
            box_array = np.array(box)
            masks, scores, _ = self.predictor.predict(
                box=box_array,
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            # Add to combined mask
            combined_mask = combined_mask | mask
            
            masked_image = image.copy()
            masked_image[~mask] = 0
            
            results.append((label, mask, masked_image))
            print(f"[Segmentor] '{label}' — {mask.sum()} pixels "
                  f"(confidence: {scores[best_idx]:.3f})")
        
        print(f"[Segmentor] Combined mask — {combined_mask.sum()} total pixels")
        return combined_mask, results


def download_sam():
    """Download SAM checkpoint if not present."""
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    os.makedirs("models", exist_ok=True)
    print("Downloading SAM checkpoint (~2.4GB)... this will take a while")
    urllib.request.urlretrieve(url, "models/sam_vit_h_4b8939.pth")
    print("Download complete.")
import numpy as np
import torch
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import pipeline
import cv2

class DepthEstimator:
    def __init__(self):
        """
        Initialize Depth-Anything-V2 for monocular depth estimation.
        Downloads model automatically on first run (~400MB).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Depth] Using device: {self.device}")
        
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if self.device == "cuda" else -1
        )
        print("[Depth] Depth-Anything-V2 loaded successfully")

    def estimate(self, image_path):
        """
        Estimate per-pixel depth from a single RGB image.
        
        Args:
            image_path: path to image file
            
        Returns:
            depth_map: numpy array (H, W) with relative depth values
            image_size: (H, W) tuple
        """
        image = Image.open(image_path).convert("RGB")
        H, W = np.array(image).shape[:2]
        
        # Run depth estimation
        result = self.pipe(image)
        depth = np.array(result["depth"])
        
        # Resize depth map to match original image size
        depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-10 meter range (relative depth)
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)
        depth_metric = depth_normalized * 10.0
        
        print(f"[Depth] Depth map generated — shape: {depth_metric.shape}, "
              f"range: {depth_metric.min():.2f}m to {depth_metric.max():.2f}m")
        
        return depth_metric, (H, W)

    def get_camera_intrinsics(self, image_path):
        """
        Extract camera intrinsics from image EXIF data.
        
        THIS IS YOUR KEY ADDITION over the original notebook.
        Original used hardcoded fx=fy=500 which is wrong for most cameras.
        We extract real focal length from EXIF metadata.
        
        Camera intrinsics:
        - fx, fy: focal lengths in pixels (how "zoomed in" the camera is)
        - cx, cy: principal point (usually image center)
        
        Back-projection formula:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth value
        
        Args:
            image_path: path to image file
            
        Returns:
            fx, fy, cx, cy: camera intrinsic parameters
        """
        image = Image.open(image_path)
        H, W = np.array(image).shape[:2]
        
        # Default: heuristic fallback if no EXIF
        # fx = max(H, W) is a reasonable estimate for standard cameras
        fx = fy = float(max(H, W))
        cx, cy = W / 2.0, H / 2.0
        
        # Try to extract real focal length from EXIF
        try:
            exif_data = image._getexif()
            if exif_data:
                exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
                
                if "FocalLength" in exif:
                    focal_mm = float(exif["FocalLength"])
                    
                    # Convert focal length from mm to pixels
                    # Using sensor width heuristic: sensor_width ≈ 36mm for full frame
                    # pixels_per_mm = image_width / sensor_width_mm
                    sensor_width_mm = 36.0  # standard full-frame assumption
                    
                    if "FocalPlaneXResolution" in exif and "FocalPlaneYResolution" in exif:
                        # More accurate: use focal plane resolution if available
                        x_res = float(exif["FocalPlaneXResolution"])
                        fx = focal_mm * x_res / 25.4
                        y_res = float(exif["FocalPlaneYResolution"])
                        fy = focal_mm * y_res / 25.4
                    else:
                        # Fallback: estimate pixels per mm from image width
                        pixels_per_mm = W / sensor_width_mm
                        fx = fy = focal_mm * pixels_per_mm
                    
                    print(f"[Depth] EXIF focal length: {focal_mm}mm → "
                          f"fx={fx:.1f}px, fy={fy:.1f}px")
                else:
                    print(f"[Depth] No focal length in EXIF — "
                          f"using heuristic fx=fy={fx:.1f}px")
            else:
                print(f"[Depth] No EXIF data — "
                      f"using heuristic fx=fy={fx:.1f}px")
                
        except Exception as e:
            print(f"[Depth] EXIF extraction failed ({e}) — "
                  f"using heuristic fx=fy={fx:.1f}px")
        
        return fx, fy, cx, cy


def visualize_depth(depth_map):
    """Convert depth map to colorized image for visualization."""
    depth_normalized = cv2.normalize(
        depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    return depth_colored
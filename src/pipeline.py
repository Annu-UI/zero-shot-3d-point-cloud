import time
import os
import numpy as np
from src.detection import ObjectDetector
from src.segmentation import ObjectSegmentor
from src.depth import DepthEstimator
from src.projection import PointCloudProjector
from src.metrics import PointCloudMetrics

class PointCloudPipeline:
    def __init__(self):
        """
        End-to-end pipeline connecting all 5 modules.
        
        Flow:
        Image + class name
             ↓
        YOLO-World → bounding boxes
             ↓
        SAM → pixel masks
             ↓
        Depth-Anything-V2 → depth map
             ↓
        Back-projection → 3D points
             ↓
        Metrics → quality report
             ↓
        Output: .ply + .html + metrics.json
        """
        print("\n[Pipeline] Initializing all modules...")
        print("-" * 50)
        
        self.detector = ObjectDetector()
        self.segmentor = ObjectSegmentor()
        self.depth_estimator = DepthEstimator()
        self.projector = PointCloudProjector()
        self.metrics = PointCloudMetrics()
        
        print("-" * 50)
        print("[Pipeline] All modules ready\n")

    def run(self, image_path, object_classes, output_dir="outputs"):
        """
        Run full pipeline on a single image.
        
        Args:
            image_path: path to input image
            object_classes: list of strings e.g. ["chair", "table"]
            output_dir: where to save outputs
            
        Returns:
            results: dict with paths to outputs and metrics
        """
        start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"[Pipeline] Processing: {image_path}")
        print(f"[Pipeline] Target objects: {object_classes}")
        print(f"{'='*50}\n")
        
        # --- Stage 1: Detection ---
        print("[Pipeline] Stage 1/5 — Object Detection")
        stage_start = time.time()
        
        boxes, scores, labels, image = self.detector.detect(
            image_path, object_classes
        )
        
        if len(boxes) == 0:
            print(f"[Pipeline] No objects found for classes: {object_classes}")
            return None
        
        detection_time = time.time() - stage_start
        print(f"[Pipeline] Detection done in {detection_time:.2f}s\n")
        
        # --- Stage 2: Segmentation ---
        print("[Pipeline] Stage 2/5 — Segmentation")
        stage_start = time.time()
        
        combined_mask, seg_results = self.segmentor.segment_all_boxes(
            image, boxes, labels
        )
        
        segmentation_time = time.time() - stage_start
        print(f"[Pipeline] Segmentation done in {segmentation_time:.2f}s\n")
        
        # --- Stage 3: Depth Estimation ---
        print("[Pipeline] Stage 3/5 — Depth Estimation")
        stage_start = time.time()
        
        depth_map, (H, W) = self.depth_estimator.estimate(image_path)
        fx, fy, cx, cy = self.depth_estimator.get_camera_intrinsics(image_path)
        
        depth_time = time.time() - stage_start
        print(f"[Pipeline] Depth estimation done in {depth_time:.2f}s\n")
        
        # --- Stage 4: 3D Projection ---
        print("[Pipeline] Stage 4/5 — 3D Back-Projection")
        stage_start = time.time()
        
        pcd, points, colors = self.projector.project(
            image, combined_mask, depth_map, fx, fy, cx, cy
        )
        
        # Track point count before denoising for metrics
        points_before = combined_mask.sum()
        points_after = len(pcd.points)
        
        projection_time = time.time() - stage_start
        print(f"[Pipeline] Projection done in {projection_time:.2f}s\n")
        
        # --- Stage 5: Metrics ---
        print("[Pipeline] Stage 5/5 — Computing Metrics")
        
        # Use first detected box for bbox metrics
        bbox = boxes[0]
        
        metrics = self.metrics.compute(
            mask=combined_mask,
            points_before_denoise=points_before,
            points_after_denoise=points_after,
            depth_map=depth_map,
            bbox=bbox
        )
        
        # Add timing info to metrics
        total_time = time.time() - start_time
        metrics["total_time_seconds"] = round(total_time, 2)
        metrics["detection_time"] = round(detection_time, 2)
        metrics["segmentation_time"] = round(segmentation_time, 2)
        metrics["depth_time"] = round(depth_time, 2)
        metrics["projection_time"] = round(projection_time, 2)
        metrics["image_path"] = image_path
        metrics["object_classes"] = object_classes
        metrics["objects_detected"] = labels
        
        # --- Save Outputs ---
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        ply_path = f"{output_dir}/{base_name}_pointcloud.ply"
        html_path = f"{output_dir}/{base_name}_pointcloud.html"
        metrics_path = f"{output_dir}/metrics.json"
        
        self.projector.save_ply(pcd, ply_path)
        self.projector.save_html(pcd, html_path)
        self.metrics.save(metrics, metrics_path)
        
        print(f"\n[Pipeline] ✓ Complete in {total_time:.2f}s")
        print(f"[Pipeline] ✓ PLY saved: {ply_path}")
        print(f"[Pipeline] ✓ HTML saved: {html_path}")
        print(f"[Pipeline] ✓ Metrics saved: {metrics_path}")
        
        return {
            "pcd": pcd,
            "ply_path": ply_path,
            "html_path": html_path,
            "metrics_path": metrics_path,
            "metrics": metrics,
            "labels_detected": labels,
            "image": image
        }


def run_quick_test(image_path, classes):
    """
    Quick test function — run from terminal to verify pipeline works.
    
    Usage:
        python -c "from src.pipeline import run_quick_test; 
                   run_quick_test('data/images/test.jpg', ['chair'])"
    """
    pipeline = PointCloudPipeline()
    results = pipeline.run(image_path, classes)
    
    if results:
        print(f"\nSuccess! Open this file in browser to see 3D result:")
        print(f"  {results['html_path']}")
        print(f"\nKey metrics:")
        print(f"  Coverage: {results['metrics']['mask_coverage_pct']}%")
        print(f"  Quality:  {results['metrics']['quality_score']}/100")
    else:
        print("Pipeline failed — no objects detected")
    
    return results
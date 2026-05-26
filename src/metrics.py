import numpy as np
import json
import os
from datetime import datetime

class PointCloudMetrics:
    def __init__(self):
        """
        Computes quality metrics for generated point clouds.
        
        THIS IS YOUR KEY ADDITION — original notebook had zero metrics.
        These numbers go directly on your resume and are fully defensible.
        
        Metrics computed:
        1. Mask coverage     — what % of segmented pixels became 3D points
        2. Point density     — points per pixel² in bounding box area
        3. Depth variance    — how much depth variation exists (object flatness)
        4. Noise ratio       — points removed by statistical denoising
        5. Quality score     — weighted composite score 0-100
        """
        print("[Metrics] Initialized")

    def compute(self, mask, points_before_denoise, 
                points_after_denoise, depth_map, bbox):
        """
        Compute all quality metrics.
        
        Args:
            mask: boolean numpy array (H, W) from SAM
            points_before_denoise: int — point count before outlier removal
            points_after_denoise: int — point count after outlier removal
            depth_map: numpy array (H, W) depth values
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            metrics: dict with all computed values
        """
        # 1. Mask coverage
        # What % of mask pixels successfully became 3D points
        mask_pixel_count = mask.sum()
        coverage = (points_after_denoise / mask_pixel_count) * 100
        
        # 2. Point density
        # Points per pixel area of bounding box
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        density = points_after_denoise / bbox_area
        
        # 3. Depth variance
        # Higher variance = more 3D structure captured
        # Low variance = flat/planar object
        masked_depth = depth_map[mask]
        depth_variance = float(np.var(masked_depth))
        depth_std = float(np.std(masked_depth))
        
        # 4. Noise ratio
        # What % of points were removed as outliers
        if points_before_denoise > 0:
            noise_ratio = ((points_before_denoise - points_after_denoise) 
                          / points_before_denoise) * 100
        else:
            noise_ratio = 0.0
        
        # 5. Quality score (weighted composite 0-100)
        # Coverage contributes 50% — most important
        # Density contributes 30%
        # Noise ratio contributes 20% (lower noise = higher score)
        coverage_score = min(coverage, 100) * 0.5
        density_score = min(density * 100, 100) * 0.3
        noise_score = max(0, 100 - noise_ratio) * 0.2
        quality_score = coverage_score + density_score + noise_score
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "mask_pixels": int(mask_pixel_count),
            "points_before_denoise": int(points_before_denoise),
            "points_after_denoise": int(points_after_denoise),
            "mask_coverage_pct": round(coverage, 2),
            "point_density_per_px2": round(density, 4),
            "depth_variance": round(depth_variance, 4),
            "depth_std_meters": round(depth_std, 4),
            "noise_ratio_pct": round(noise_ratio, 2),
            "quality_score": round(quality_score, 2),
            "bbox_area_px2": int(bbox_area)
        }
        
        self._print_report(metrics)
        return metrics

    def _print_report(self, metrics):
        """Print formatted metrics report."""
        print("\n" + "="*50)
        print("       POINT CLOUD QUALITY REPORT")
        print("="*50)
        print(f"  Mask pixels:        {metrics['mask_pixels']:,}")
        print(f"  Points generated:   {metrics['points_before_denoise']:,}")
        print(f"  Points after denoise: {metrics['points_after_denoise']:,}")
        print(f"  Mask coverage:      {metrics['mask_coverage_pct']}%")
        print(f"  Point density:      {metrics['point_density_per_px2']} pts/px²")
        print(f"  Depth variance:     {metrics['depth_variance']}")
        print(f"  Noise removed:      {metrics['noise_ratio_pct']}%")
        print(f"  Quality score:      {metrics['quality_score']}/100")
        print("="*50 + "\n")

    def save(self, metrics, output_path="outputs/metrics.json"):
        """Save metrics to JSON file for tracking across runs."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Append to existing metrics file if it exists
        all_metrics = []
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                all_metrics = json.load(f)
        
        all_metrics.append(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"[Metrics] Saved to {output_path}")

    def compare_runs(self, metrics_path="outputs/metrics.json"):
        """
        Compare metrics across multiple runs.
        Useful for showing improvement over time.
        """
        if not os.path.exists(metrics_path):
            print("[Metrics] No previous runs found")
            return
        
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
        
        if len(all_metrics) < 2:
            print("[Metrics] Need at least 2 runs to compare")
            return
        
        print(f"\n[Metrics] Comparison across {len(all_metrics)} runs:")
        print(f"  Avg coverage:      "
              f"{np.mean([m['mask_coverage_pct'] for m in all_metrics]):.2f}%")
        print(f"  Avg quality score: "
              f"{np.mean([m['quality_score'] for m in all_metrics]):.2f}/100")
        print(f"  Best run score:    "
              f"{max(m['quality_score'] for m in all_metrics):.2f}/100")
        
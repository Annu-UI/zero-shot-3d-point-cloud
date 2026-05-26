import numpy as np
import open3d as o3d
import cv2
import os

class PointCloudProjector:
    def __init__(self):
        """
        Handles 3D back-projection from 2D pixels + depth map.
        
        Core math:
        Given a pixel (u, v) with depth Z and camera intrinsics (fx, fy, cx, cy):
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z (depth value directly)
        
        This converts 2D image coordinates to 3D world coordinates.
        """
        print("[Projector] Initialized")

    def project(self, image, mask, depth_map, fx, fy, cx, cy):
        """
        Back-project masked pixels into 3D point cloud.
        
        Args:
            image: RGB numpy array (H, W, 3)
            mask: boolean numpy array (H, W) from SAM
            depth_map: float numpy array (H, W) in meters
            fx, fy: focal lengths in pixels
            cx, cy: principal point (image center)
            
        Returns:
            pcd: Open3D point cloud object
            points: numpy array (N, 3) of 3D coordinates
            colors: numpy array (N, 3) of RGB colors normalized 0-1
        """
        H, W = depth_map.shape
        
        # Apply bilateral filter before projection
        # This smooths depth while preserving edges — key for clean point clouds
        depth_smooth = cv2.bilateralFilter(
            depth_map.astype(np.float32),
            d=9,           # filter diameter
            sigmaColor=75, # color space sigma
            sigmaSpace=75  # coordinate space sigma
        )
        
        # Get pixel coordinates where mask is True
        v_coords, u_coords = np.where(mask)
        
        # Get depth values at masked pixels
        Z = depth_smooth[v_coords, u_coords]
        
        # Filter out zero or very small depth values
        valid = Z > 0.1
        u_valid = u_coords[valid]
        v_valid = v_coords[valid]
        Z_valid = Z[valid]
        
        # Back-projection math — this is the core formula
        X = (u_valid - cx) * Z_valid / fx
        Y = (v_valid - cy) * Z_valid / fy
        
        # Stack into (N, 3) points array
        points = np.stack([X, Y, Z_valid], axis=1)
        
        # Get colors for each point from original image
        colors = image[v_valid, u_valid] / 255.0  # normalize to 0-1
        
        print(f"[Projector] Generated {len(points)} 3D points "
              f"from {mask.sum()} masked pixels")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Statistical outlier removal — removes noise points
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,  # consider 20 nearest neighbors
            std_ratio=2.0     # remove points 2 std devs from mean
        )
        
        print(f"[Projector] After denoising: "
              f"{len(pcd.points)} points remaining")
        
        return pcd, points, colors

    def save_ply(self, pcd, output_path="outputs/point_cloud.ply"):
        """Save point cloud to .ply file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"[Projector] Saved to {output_path}")

    def save_html(self, pcd, output_path="outputs/point_cloud.html"):
        """
        Save interactive HTML visualization.
        User can rotate/zoom in browser — no special software needed.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Convert colors to hex for plotly
        colors_255 = (colors * 255).astype(int)
        color_strings = [
            f'rgb({r},{g},{b})' 
            for r, g, b in colors_255
        ]
        
        # Generate standalone HTML with plotly
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D Point Cloud</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ margin: 0; background: #1a1a1a; }}
        #plot {{ width: 100vw; height: 100vh; }}
    </style>
</head>
<body>
<div id="plot"></div>
<script>
    var trace = {{
        x: {points[:, 0].tolist()},
        y: {points[:, 1].tolist()},
        z: {points[:, 2].tolist()},
        mode: 'markers',
        type: 'scatter3d',
        marker: {{
            size: 2,
            color: {color_strings},
            opacity: 0.8
        }}
    }};
    var layout = {{
        paper_bgcolor: '#1a1a1a',
        scene: {{
            bgcolor: '#1a1a1a',
            xaxis: {{color: 'white'}},
            yaxis: {{color: 'white'}},
            zaxis: {{color: 'white'}}
        }},
        margin: {{l:0, r:0, t:0, b:0}}
    }};
    Plotly.newPlot('plot', [trace], layout);
</script>
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html_content)
        print(f"[Projector] Interactive HTML saved to {output_path}")
        return output_path
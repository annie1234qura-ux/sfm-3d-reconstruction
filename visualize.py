import open3d as o3d
import sys
import os

def visualize_ply(path="output/reconstruction.ply"):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        print("Run sfm.py first.")
        sys.exit(1)

    print(f"Loading: {path}")
    pcd = o3d.io.read_point_cloud(path)
    print(f"Points: {len(pcd.points)}")

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.2, 0.6, 1.0])

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After cleanup: {len(pcd.points)} points")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="SfM 3D Reconstruction",
        width=1280,
        height=720
    )

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/reconstruction.ply"
    visualize_ply(path)
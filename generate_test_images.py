import cv2
import numpy as np
import os

def project_points(points_3d, K, R, t):
    pts, _ = cv2.projectPoints(points_3d.astype(np.float32),
                                cv2.Rodrigues(R)[0], t, K, None)
    return pts.reshape(-1, 2)

def draw_box(img, pts2d, color=(0, 200, 255), thickness=2):
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    pts = pts2d.astype(int)
    for i, j in edges:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, thickness)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (255, 100, 0), -1)

def generate(output_dir="images", n_views=6):
    os.makedirs(output_dir, exist_ok=True)
    W, H = 800, 600
    focal = max(W, H)
    K = np.array([[focal, 0, W/2],
                  [0, focal, H/2],
                  [0,     0,   1]], dtype=np.float64)

    box = np.array([
        [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
        [-1,-1, 1], [1,-1, 1], [1,1, 1], [-1,1, 1]
    ], dtype=np.float32) * 0.5

    np.random.seed(42)
    cloud = (np.random.rand(60, 3) - 0.5).astype(np.float32)
    angles = np.linspace(0, np.pi * 1.2, n_views)

    print(f"Generating {n_views} synthetic views...")
    for i, angle in enumerate(angles):
        cx = 4.0 * np.sin(angle)
        cz = 4.0 * np.cos(angle)
        cam_pos = np.array([cx, 0.5, cz])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, -up, forward])
        t = (-R @ cam_pos).reshape(3, 1)

        img = np.ones((H, W, 3), dtype=np.uint8) * 30

        for gx in np.linspace(-2, 2, 9):
            p1 = project_points(np.array([[gx, -0.5, -2]]), K, R, t)
            p2 = project_points(np.array([[gx, -0.5,  2]]), K, R, t)
            cv2.line(img, tuple(p1[0].astype(int)), tuple(p2[0].astype(int)), (50,50,50), 1)
        for gz in np.linspace(-2, 2, 9):
            p1 = project_points(np.array([[-2, -0.5, gz]]), K, R, t)
            p2 = project_points(np.array([[ 2, -0.5, gz]]), K, R, t)
            cv2.line(img, tuple(p1[0].astype(int)), tuple(p2[0].astype(int)), (50,50,50), 1)

        c2d = project_points(cloud, K, R, t)
        for p in c2d.astype(int):
            if 0 <= p[0] < W and 0 <= p[1] < H:
                cv2.circle(img, tuple(p), 2, (100, 255, 100), -1)

        b2d = project_points(box, K, R, t)
        draw_box(img, b2d)

        cv2.putText(img, f"View {i+1} | angle={np.degrees(angle):.0f}deg",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        path = os.path.join(output_dir, f"view_{i+1:02d}.jpg")
        cv2.imwrite(path, img)
        print(f"  Saved: {path}")

    print(f"\nDone! Run: python sfm.py")

if __name__ == "__main__":
    generate()
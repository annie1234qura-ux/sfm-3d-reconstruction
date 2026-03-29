"""
Sparse 3D Reconstruction from Images - Structure from Motion (SfM)
"""

import cv2
import numpy as np
import os
import glob


def load_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                   glob.glob(os.path.join(folder, "*.png")))
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in '{folder}'. Found: {len(paths)}")
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append((os.path.basename(p), img))
            print(f"  Loaded: {os.path.basename(p)}  {img.shape[1]}x{img.shape[0]}")
    return images


def detect_and_match(img1, img2, ratio=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < ratio * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, good, kp1, kp2


def get_camera_matrix(img):
    h, w = img.shape[:2]
    focal = max(w, h)
    K = np.array([
        [focal,     0,  w / 2],
        [    0, focal,  h / 2],
        [    0,     0,      1]
    ], dtype=np.float64)
    return K


def estimate_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC,
                                   prob=0.999,
                                   threshold=1.0)
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inliers = mask_pose.ravel() > 0
    return R, t, inliers


def triangulate(pts1, pts2, K, R1, t1, R2, t2):
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])
    pts4d = cv2.triangulatePoints(P1, P2,
                                  pts1.T.astype(np.float32),
                                  pts2.T.astype(np.float32))
    pts3d = (pts4d[:3] / pts4d[3]).T
    valid = (pts4d[3] != 0) & (pts3d[:, 2] > 0)
    return pts3d[valid]


def save_ply(points, filename):
    points = np.array(points)
    mean = np.mean(points, axis=0)
    std  = np.std(points, axis=0)
    mask = np.all(np.abs(points - mean) < 3 * std, axis=1)
    points = points[mask]
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"\n  Saved {len(points)} points -> {filename}")


def run_sfm(image_folder="images", output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    print("\n" + "="*55)
    print("   Structure from Motion — Sparse 3D Reconstruction")
    print("="*55)

    print("\n[1/5] Loading images...")
    images = load_images(image_folder)
    print(f"  Total images: {len(images)}")

    all_points_3d = []
    pairs = list(zip(images[:-1], images[1:]))

    print("\n[2/5] Detecting features & matching pairs...")
    for (name1, img1), (name2, img2) in pairs:
        print(f"\n  Pair: {name1} <-> {name2}")
        pts1, pts2, good_matches, kp1, kp2 = detect_and_match(img1, img2)
        print(f"  Good matches: {len(good_matches)}")

        if len(good_matches) < 50:
            print("  Too few matches, skipping pair.")
            continue

        match_img = cv2.drawMatches(img1, kp1, img2, kp2,
                                    good_matches[:50], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        match_path = os.path.join(output_folder,
                                  f"matches_{name1.split('.')[0]}_{name2.split('.')[0]}.jpg")
        cv2.imwrite(match_path, match_img)

        K = get_camera_matrix(img1)

        print("[3/5] Estimating camera pose...")
        R, t, inliers = estimate_pose(pts1, pts2, K)
        if R is None:
            print("  Could not estimate pose, skipping.")
            continue
        print(f"  Inliers after RANSAC: {inliers.sum()}")

        print("[4/5] Triangulating 3D points...")
        R1 = np.eye(3);   t1 = np.zeros((3, 1))
        R2 = R;           t2 = t
        pts3d = triangulate(pts1[inliers], pts2[inliers], K, R1, t1, R2, t2)
        print(f"  Triangulated {len(pts3d)} 3D points")
        all_points_3d.append(pts3d)

    print("\n[5/5] Saving point cloud...")
    if not all_points_3d:
        print("  No 3D points reconstructed. Check your images.")
        return

    combined = np.vstack(all_points_3d)
    out_path = os.path.join(output_folder, "reconstruction.ply")
    save_ply(combined, out_path)

    print("\n" + "="*55)
    print("  Done! Open output/reconstruction.ply in MeshLab")
    print("="*55 + "\n")


if __name__ == "__main__":
    run_sfm(image_folder="images", output_folder="output")
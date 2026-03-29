# Sparse 3D Reconstruction — Structure from Motion (SfM)

## What is this project?
This project takes multiple photos of the same object from different angles 
and automatically creates a 3D point cloud from them.

## How to run
1. Install dependencies:
pip install opencv-contrib-python numpy open3d flask

2. Generate test images:
python generate_test_images.py

3. Run the web app:
python app.py

4. Open browser:
http://localhost:5000

## Technologies Used
- Python
- OpenCV (SIFT, RANSAC, Triangulation)
- Flask (Web App)
- Three.js (3D Viewer)
- NumPy

## Course
Computer Vision 

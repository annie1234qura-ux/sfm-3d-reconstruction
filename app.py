"""
app.py — Web interface for SfM 3D Reconstruction
Supports: Images, Video files, Live Camera
"""

from flask import Flask, render_template_string, request, jsonify, Response
import cv2
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename
import threading
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('output', exist_ok=True)

# Global camera object
camera = None
camera_lock = threading.Lock()
captured_frames = []
is_capturing = False


# ─── SfM Pipeline ───────────────────────────────────────────

def detect_and_match(img1, img2, ratio=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None, None, []
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        raw_matches = flann.knnMatch(des1, des2, k=2)
    except:
        return None, None, []
    good = [m for m, n in raw_matches if m.distance < ratio * n.distance]
    if not good:
        return None, None, []
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, good

def get_camera_matrix(img):
    h, w = img.shape[:2]
    focal = max(w, h)
    return np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)

def estimate_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inliers = mask_pose.ravel() > 0
    return R, t, inliers

def triangulate(pts1, pts2, K, R1, t1, R2, t2):
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float32), pts2.T.astype(np.float32))
    pts3d = (pts4d[:3] / pts4d[3]).T
    valid = (pts4d[3] != 0) & (pts3d[:, 2] > 0)
    return pts3d[valid]

def run_sfm_on_images(images):
    if len(images) < 2:
        return [], "Need at least 2 images"

    all_points = []
    log = []
    for i, (img1, img2) in enumerate(zip(images[:-1], images[1:])):
        pts1, pts2, good = detect_and_match(img1, img2)
        if pts1 is None or len(good) < 30:
            log.append(f"Pair {i+1}: {len(good) if good else 0} matches - skipping")
            continue
        K = get_camera_matrix(img1)
        R, t, inliers = estimate_pose(pts1, pts2, K)
        if R is None:
            log.append(f"Pair {i+1}: pose failed")
            continue
        pts3d = triangulate(pts1[inliers], pts2[inliers], K,
                            np.eye(3), np.zeros((3,1)), R, t)
        all_points.append(pts3d)
        log.append(f"Pair {i+1}: {len(good)} matches → {len(pts3d)} 3D points")

    if not all_points:
        return [], "Reconstruction failed. Need more photos with overlap."

    combined = np.vstack(all_points)
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    mask = np.all(np.abs(combined - mean) < 3 * std, axis=1)
    return combined[mask].tolist(), "\n".join(log)


# ─── Video Processing ────────────────────────────────────────

def extract_frames_from_video(video_path, max_frames=20):
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 2:
        cap.release()
        return []
    
    # Pick evenly spaced frames
    step = max(1, total_frames // max_frames)
    frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Resize if too large
            h, w = frame.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            frames.append(frame)
        frame_idx += 1
        if len(frames) >= max_frames:
            break
    
    cap.release()
    return frames


# ─── Camera Streaming ────────────────────────────────────────

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    """Stream live camera frames to browser."""
    cam = get_camera()
    while True:
        with camera_lock:
            success, frame = cam.read()
        if not success:
            break
        # Add overlay text
        cv2.putText(frame, "LIVE CAMERA - Press CAPTURE to take photo", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {len(captured_frames)} frames", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30fps


# ─── Routes ─────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    """Handle image upload reconstruction."""
    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({'error': 'Need at least 2 images'})
    
    images = []
    for f in files:
        fname = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, fname)
        f.save(path)
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)))
            images.append(img)
        try: os.remove(path)
        except: pass
    
    points, log = run_sfm_on_images(images)
    if not points:
        return jsonify({'error': log})
    return jsonify({'points': points, 'log': log})


@app.route('/reconstruct_video', methods=['POST'])
def reconstruct_video():
    """Handle video file reconstruction."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'})
    
    f = request.files['video']
    fname = secure_filename(f.filename)
    path = os.path.join(UPLOAD_FOLDER, fname)
    f.save(path)
    
    log_msgs = [f"Video received: {fname}"]
    log_msgs.append("Extracting frames from video...")
    
    frames = extract_frames_from_video(path, max_frames=20)
    try: os.remove(path)
    except: pass
    
    if len(frames) < 2:
        return jsonify({'error': 'Could not extract enough frames from video'})
    
    log_msgs.append(f"Extracted {len(frames)} frames from video")
    points, sfm_log = run_sfm_on_images(frames)
    
    if not points:
        return jsonify({'error': sfm_log})
    
    return jsonify({'points': points, 'log': "\n".join(log_msgs) + "\n" + sfm_log})


@app.route('/video_feed')
def video_feed():
    """Live camera stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Capture a frame from live camera."""
    global captured_frames
    cam = get_camera()
    with camera_lock:
        success, frame = cam.read()
    if not success:
        return jsonify({'error': 'Could not capture frame', 'count': len(captured_frames)})
    
    h, w = frame.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    
    captured_frames.append(frame)
    return jsonify({'success': True, 'count': len(captured_frames)})

@app.route('/reconstruct_camera', methods=['POST'])
def reconstruct_camera():
    """Run SfM on captured camera frames."""
    global captured_frames
    if len(captured_frames) < 2:
        return jsonify({'error': f'Need at least 2 captured frames. You have {len(captured_frames)}'})
    
    points, log = run_sfm_on_images(captured_frames)
    captured_frames = []  # Reset after reconstruction
    
    if not points:
        return jsonify({'error': log})
    return jsonify({'points': points, 'log': f"Used {len(captured_frames)} camera frames\n" + log})

@app.route('/reset_camera', methods=['POST'])
def reset_camera():
    """Reset captured frames."""
    global captured_frames
    captured_frames = []
    return jsonify({'success': True, 'count': 0})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Release camera."""
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'success': True})


# ─── HTML ────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>SfM 3D Reconstruction</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #080c14; color: #c8dff0; font-family: Arial, sans-serif; min-height: 100vh; }
body::before { content:''; position:fixed; inset:0; background-image: linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px), linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px); background-size:40px 40px; pointer-events:none; z-index:0; }
h1 { text-align:center; padding:24px; color:#00d4ff; font-size:1.6rem; letter-spacing:3px; position:relative; z-index:1; }
.layout { display:flex; gap:16px; padding:0 20px 20px; flex-wrap:wrap; position:relative; z-index:1; }
.left { width:360px; }
.right { flex:1; min-width:300px; }

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: #120f1f;
    color: #f3ecff;
    font-family: Arial, sans-serif;
    min-height: 100vh;
}

body::before {
    content:'';
    position:fixed;
    inset:0;
    background-image:
        linear-gradient(rgba(216,180,254,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(216,180,254,0.04) 1px, transparent 1px);
    background-size:40px 40px;
    pointer-events:none;
    z-index:0;
}

h1 {
    text-align:center;
    padding:24px;
    color:#d8b4fe;
    font-size:1.6rem;
    letter-spacing:3px;
    position:relative;
    z-index:1;
    text-shadow:0 0 15px rgba(216,180,254,0.5);
}

.layout {
    display:flex;
    gap:16px;
    padding:0 20px 20px;
    flex-wrap:wrap;
    position:relative;
    z-index:1;
}

.left { width:360px; }
.right { flex:1; min-width:300px; }

/* Tabs */

.tabs {
    display:flex;
    gap:4px;
    margin-bottom:12px;
}

.tab {
    flex:1;
    padding:10px;
    background:#1b162c;
    border:1px solid #4c3d6d;
    color:#bba4e8;
    font-size:0.8rem;
    letter-spacing:1px;
    cursor:pointer;
    text-align:center;
    border-radius:4px;
    transition:all 0.25s;
}

.tab.active {
    background:#34264f;
    color:#f3ecff;
    border-color:#d8b4fe;
}

.tab:hover {
    color:#e9d5ff;
}

/* Panels */

.panel { display:none; }
.panel.active { display:block; }

.box {
    background:#1b162c;
    border:1px solid #4c3d6d;
    border-radius:6px;
    padding:16px;
    margin-bottom:12px;
    box-shadow:0 0 20px rgba(216,180,254,0.08);
}

.box h2 {
    color:#d8b4fe;
    font-size:0.75rem;
    letter-spacing:2px;
    margin-bottom:12px;
}

/* Upload */

.upload-btn {
    display:block;
    width:100%;
    padding:30px 16px;
    background:#171122;
    border:2px dashed #4c3d6d;
    border-radius:6px;
    color:#c8b7ea;
    text-align:center;
    cursor:pointer;
    font-size:0.85rem;
    transition:all 0.25s;
    margin-bottom:10px;
}

.upload-btn:hover {
    border-color:#d8b4fe;
    color:#f3ecff;
    box-shadow:0 0 15px rgba(216,180,254,0.2);
}

.upload-btn .icon {
    font-size:2rem;
    display:block;
    margin-bottom:6px;
}

input[type=file] {
    display:none;
}

#thumbGrid {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:5px;
    margin-top:10px;
}

#thumbGrid img {
    width:100%;
    aspect-ratio:1;
    object-fit:cover;
    border-radius:3px;
    border:1px solid #4c3d6d;
}

#videoThumb {
    width:100%;
    border-radius:4px;
    border:1px solid #4c3d6d;
    margin-top:10px;
    display:none;
}

/* Camera */

#cameraFeed {
    width:100%;
    border-radius:4px;
    border:1px solid #4c3d6d;
    display:none;
}

.cam-controls {
    display:flex;
    gap:8px;
    margin-top:8px;
    flex-wrap:wrap;
}

/* Buttons */

.btn {
    padding:10px 14px;
    background:transparent;
    border:1px solid #d8b4fe;
    color:#d8b4fe;
    font-size:0.75rem;
    letter-spacing:1px;
    cursor:pointer;
    border-radius:3px;
    transition:all 0.25s;
}

.btn:hover {
    background:#d8b4fe;
    color:#120f1f;
    box-shadow:0 0 15px rgba(216,180,254,0.5);
}

.btn:disabled {
    opacity:0.3;
    cursor:not-allowed;
}

.btn:disabled:hover {
    background:transparent;
    color:#d8b4fe;
}

.btn-full {
    width:100%;
    margin-top:10px;
    padding:12px;
}

.btn-red {
    border-color:#ff6b9f;
    color:#ff6b9f;
}

.btn-red:hover {
    background:#ff6b9f;
    color:white;
}

.btn-green {
    border-color:#86efac;
    color:#86efac;
}

.btn-green:hover {
    background:#86efac;
    color:#120f1f;
}

/* Logs */

#log,
#videoLog,
#camLog {
    font-family:monospace;
    font-size:0.72rem;
    color:#c8b7ea;
    white-space:pre-wrap;
    line-height:1.8;
    padding:10px;
    background:#120f1f;
    border-radius:3px;
    min-height:80px;
    margin-top:10px;
}

.ok { color:#86efac; }
.err { color:#ff8fa3; }
.info { color:#d8b4fe; }

#fileInfo,
#videoInfo {
    color:#c8b7ea;
    font-size:0.75rem;
    margin-top:6px;
    text-align:center;
}

.capture-count {
    font-family:monospace;
    font-size:1.2rem;
    color:#d8b4fe;
    text-align:center;
    margin:8px 0;
}

/* Viewer */

#viewer {
    width:100%;
    aspect-ratio:16/10;
    background:#0f0b19;
    border-radius:6px;
    position:relative;
    overflow:hidden;
}

#placeholder {
    position:absolute;
    inset:0;
    display:flex;
    align-items:center;
    justify-content:center;
    color:#7b639f;
    font-size:0.8rem;
    letter-spacing:2px;
    flex-direction:column;
    gap:10px;
}

.spin {
    width:32px;
    height:32px;
    border:2px solid #4c3d6d;
    border-top-color:#d8b4fe;
    border-radius:50%;
    animation:sp 0.8s linear infinite;
    display:none;
}

@keyframes sp {
    to { transform:rotate(360deg); }
}

.hint {
    color:#8f7ab4;
    font-size:0.65rem;
    text-align:center;
    padding:6px;
    letter-spacing:1px;
}

.stats {
    display:flex;
    gap:8px;
    margin-top:10px;
}

.stat {
    flex:1;
    background:#120f1f;
    border:1px solid #4c3d6d;
    border-radius:3px;
    padding:8px;
    text-align:center;
}

.stat-n {
    color:#d8b4fe;
    font-size:1.2rem;
    font-family:monospace;
    display:block;
}

.stat-l {
    color:#c8b7ea;
    font-size:0.6rem;
    letter-spacing:1px;
}
</style>
</head>
<body>
<h1>STRUCTURE FROM MOTION — 3D RECONSTRUCTION</h1>
<div class="layout">
  <div class="left">

    <!-- Tabs -->
    <div class="tabs">
      <div class="tab active" onclick="switchTab('images')">📷 IMAGES</div>
      <div class="tab" onclick="switchTab('video')">🎬 VIDEO</div>
      <div class="tab" onclick="switchTab('camera')">📹 LIVE CAM</div>
    </div>

    <!-- Images Panel -->
    <div class="panel active" id="panel-images">
      <div class="box">
        <h2>UPLOAD PHOTOS</h2>
        <label class="upload-btn" for="fileInput">
          <span class="icon">📂</span>
          Click to select photos
          <small style="display:block;margin-top:4px;font-size:0.72rem">Select multiple with Ctrl+Click • Min 5 photos</small>
        </label>
        <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.png">
        <div id="thumbGrid"></div>
        <div id="fileInfo">No files selected</div>
        <button class="btn btn-full" id="runImgBtn" disabled onclick="runImages()">▶ RUN RECONSTRUCTION</button>
        <div id="log">Waiting for images...</div>
      </div>
    </div>

    <!-- Video Panel -->
    <div class="panel" id="panel-video">
      <div class="box">
        <h2>UPLOAD VIDEO</h2>
        <label class="upload-btn" for="videoInput">
          <span class="icon">🎬</span>
          Click to select video file
          <small style="display:block;margin-top:4px;font-size:0.72rem">MP4, AVI, MOV supported • Walk around object slowly</small>
        </label>
        <input type="file" id="videoInput" accept="video/*">
        <video id="videoThumb" controls></video>
        <div id="videoInfo" style="color:#4a6b85;font-size:0.75rem;margin-top:6px;text-align:center;">No video selected</div>
        <button class="btn btn-full" id="runVideoBtn" disabled onclick="runVideo()">▶ RUN RECONSTRUCTION</button>
        <div id="videoLog">Waiting for video...</div>
      </div>
    </div>

    <!-- Camera Panel -->
    <div class="panel" id="panel-camera">
      <div class="box">
        <h2>LIVE CAMERA</h2>
        <img id="cameraFeed" src="" alt="Camera feed">
        <div class="capture-count" id="captureCount">0 frames captured</div>
        <div class="cam-controls">
          <button class="btn btn-green" onclick="startCamera()">▶ START CAMERA</button>
          <button class="btn" onclick="captureFrame()">📸 CAPTURE</button>
          <button class="btn btn-red" onclick="resetFrames()">🗑 RESET</button>
          <button class="btn btn-red" onclick="stopCamera()">⏹ STOP</button>
        </div>
        <button class="btn btn-full" id="runCamBtn" disabled onclick="runCamera()">▶ RECONSTRUCT FROM CAPTURES</button>
        <div id="camLog">Start camera and capture at least 5 frames from different angles...</div>
      </div>
    </div>

  </div>

  <!-- Right: 3D Viewer -->
  <div class="right">
    <div class="box">
      <h2>3D POINT CLOUD VIEWER</h2>
      <div id="viewer">
        <div id="placeholder">
          <div class="spin" id="spin"></div>
          <span id="phText">Upload images, video, or use live camera</span>
        </div>
      </div>
      <div class="hint">LEFT DRAG = rotate &nbsp;|&nbsp; SCROLL = zoom &nbsp;|&nbsp; RIGHT DRAG = pan</div>
    </div>
    <div class="stats" id="stats" style="display:none">
      <div class="stat"><span class="stat-n" id="sPoints">0</span><span class="stat-l">3D POINTS</span></div>
      <div class="stat"><span class="stat-n" id="sImages">0</span><span class="stat-l">IMAGES</span></div>
      <div class="stat"><span class="stat-n" id="sPairs">0</span><span class="stat-l">PAIRS</span></div>
    </div>
  </div>
</div>

<script>
let files = [];
let videoFile = null;
let scene, camera3d, renderer, pc;
let drag=false, rightDrag=false, last={x:0,y:0};
let sph={t:0.5,p:1.0,r:2.5};
let camCount = 0;

// ── Tabs ──────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ['images','video','camera'][i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  if(name !== 'camera') stopCamera();
}

// ── Images ────────────────────────────────────────────────
document.getElementById('fileInput').addEventListener('change', function() {
  files = Array.from(this.files);
  const grid = document.getElementById('thumbGrid');
  grid.innerHTML = '';
  files.slice(0,12).forEach(f => {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    grid.appendChild(img);
  });
  document.getElementById('fileInfo').textContent = files.length + ' photos selected';
  document.getElementById('runImgBtn').disabled = files.length < 2;
  setLog('log', files.length + ' images ready. Click RUN.', 'info');
});

async function runImages() {
  if(files.length < 2) return;
  document.getElementById('runImgBtn').disabled = true;
  showLoading('Processing ' + files.length + ' images...');
  setLog('log', 'Uploading and processing ' + files.length + ' images...', 'info');
  const form = new FormData();
  files.forEach(f => form.append('images', f));
  try {
    const res = await fetch('/reconstruct', {method:'POST', body:form});
    const data = await res.json();
    if(data.error) { setLog('log', 'ERROR: ' + data.error, 'err'); hideLoading('FAILED'); }
    else { setLog('log', data.log + '\\n\\nSUCCESS: ' + data.points.length + ' 3D points!', 'ok'); show3D(data.points, files.length); }
  } catch(e) { setLog('log', 'ERROR: ' + e.message, 'err'); }
  document.getElementById('runImgBtn').disabled = false;
}

// ── Video ─────────────────────────────────────────────────
document.getElementById('videoInput').addEventListener('change', function() {
  videoFile = this.files[0];
  if(!videoFile) return;
  const vid = document.getElementById('videoThumb');
  vid.src = URL.createObjectURL(videoFile);
  vid.style.display = 'block';
  document.getElementById('videoInfo').textContent = videoFile.name + ' (' + (videoFile.size/1024/1024).toFixed(1) + ' MB)';
  document.getElementById('runVideoBtn').disabled = false;
  setLog('videoLog', 'Video loaded. Click RUN to extract frames and reconstruct.', 'info');
});

async function runVideo() {
  if(!videoFile) return;
  document.getElementById('runVideoBtn').disabled = true;
  showLoading('Extracting frames from video...');
  setLog('videoLog', 'Uploading video and extracting frames...\\nThis may take 1-2 minutes for large videos.', 'info');
  const form = new FormData();
  form.append('video', videoFile);
  try {
    const res = await fetch('/reconstruct_video', {method:'POST', body:form});
    const data = await res.json();
    if(data.error) { setLog('videoLog', 'ERROR: ' + data.error, 'err'); hideLoading('FAILED'); }
    else { setLog('videoLog', data.log + '\\n\\nSUCCESS: ' + data.points.length + ' 3D points!', 'ok'); show3D(data.points, 20); }
  } catch(e) { setLog('videoLog', 'ERROR: ' + e.message, 'err'); }
  document.getElementById('runVideoBtn').disabled = false;
}

// ── Camera ────────────────────────────────────────────────
function startCamera() {
  const feed = document.getElementById('cameraFeed');
  feed.src = '/video_feed?' + Date.now();
  feed.style.display = 'block';
  setLog('camLog', 'Camera started!\\nMove around your object slowly and click CAPTURE every step.\\nCapture at least 5-10 frames for good results.', 'info');
}

async function captureFrame() {
  try {
    const res = await fetch('/capture_frame', {method:'POST'});
    const data = await res.json();
    if(data.error) { setLog('camLog', 'ERROR: ' + data.error, 'err'); return; }
    camCount = data.count;
    document.getElementById('captureCount').textContent = camCount + ' frames captured';
    document.getElementById('runCamBtn').disabled = camCount < 2;
    setLog('camLog', camCount + ' frames captured.\\n' + (camCount < 5 ? 'Capture more frames for better results.' : 'Good! Click RECONSTRUCT when ready.'), camCount >= 5 ? 'ok' : 'info');
  } catch(e) { setLog('camLog', 'ERROR: ' + e.message, 'err'); }
}

async function resetFrames() {
  await fetch('/reset_camera', {method:'POST'});
  camCount = 0;
  document.getElementById('captureCount').textContent = '0 frames captured';
  document.getElementById('runCamBtn').disabled = true;
  setLog('camLog', 'Frames reset. Start capturing again.', 'info');
}

async function stopCamera() {
  await fetch('/stop_camera', {method:'POST'});
  document.getElementById('cameraFeed').style.display = 'none';
}

async function runCamera() {
  document.getElementById('runCamBtn').disabled = true;
  showLoading('Reconstructing from camera frames...');
  setLog('camLog', 'Running reconstruction on captured frames...', 'info');
  try {
    const res = await fetch('/reconstruct_camera', {method:'POST'});
    const data = await res.json();
    if(data.error) { setLog('camLog', 'ERROR: ' + data.error, 'err'); hideLoading('FAILED'); }
    else { setLog('camLog', data.log + '\\n\\nSUCCESS: ' + data.points.length + ' 3D points!', 'ok'); show3D(data.points, camCount); }
  } catch(e) { setLog('camLog', 'ERROR: ' + e.message, 'err'); }
  document.getElementById('runCamBtn').disabled = false;
}

// ── 3D Viewer ─────────────────────────────────────────────
function show3D(points, imgCount) {
  const c = document.getElementById('viewer');
  document.getElementById('placeholder').style.display = 'none';
  if(renderer) { renderer.dispose(); if(renderer.domElement.parentNode) c.removeChild(renderer.domElement); }
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050810);
  camera3d = new THREE.PerspectiveCamera(60, c.clientWidth/c.clientHeight, 0.001, 1000);
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(c.clientWidth, c.clientHeight);
  c.appendChild(renderer.domElement);
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),zs=points.map(p=>p[2]);
  const cx=(Math.min(...xs)+Math.max(...xs))/2;
  const cy=(Math.min(...ys)+Math.max(...ys))/2;
  const cz=(Math.min(...zs)+Math.max(...zs))/2;
  const span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),Math.max(...zs)-Math.min(...zs))||1;
  const minY=Math.min(...ys),maxY=Math.max(...ys);
  const geo=new THREE.BufferGeometry();
  const pos=new Float32Array(points.length*3);
  const col=new Float32Array(points.length*3);
  points.forEach((p,i)=>{
    pos[i*3]=(p[0]-cx)/span;pos[i*3+1]=(p[1]-cy)/span;pos[i*3+2]=(p[2]-cz)/span;
    const t=(p[1]-minY)/(maxY-minY+0.001);
    col[i*3]=t*0.2;col[i*3+1]=0.5+t*0.5;col[i*3+2]=1-t*0.4;
  });
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  pc=new THREE.Points(geo,new THREE.PointsMaterial({size:0.03,vertexColors:true}));
  scene.add(pc);
  scene.add(new THREE.GridHelper(2,20,0x1a2d45,0x0d1a28));
  sph={t:0.5,p:1.0,r:2.5};
  updateCam();
  renderer.domElement.addEventListener('mousedown',e=>{drag=true;rightDrag=e.button===2;last={x:e.clientX,y:e.clientY}});
  renderer.domElement.addEventListener('contextmenu',e=>e.preventDefault());
  window.addEventListener('mouseup',()=>drag=false);
  window.addEventListener('mousemove',e=>{
    if(!drag)return;
    const dx=e.clientX-last.x,dy=e.clientY-last.y;
    if(!rightDrag){sph.t-=dx*0.008;sph.p=Math.max(0.1,Math.min(3.0,sph.p+dy*0.008));}
    else{pc.position.x+=dx*0.003*sph.r;pc.position.y-=dy*0.003*sph.r;}
    last={x:e.clientX,y:e.clientY};updateCam();
  });
  renderer.domElement.addEventListener('wheel',e=>{sph.r=Math.max(0.3,Math.min(20,sph.r+e.deltaY*0.005));updateCam();});
  (function loop(){requestAnimationFrame(loop);pc.rotation.y+=0.003;renderer.render(scene,camera3d);})();
  document.getElementById('sPoints').textContent=points.length;
  document.getElementById('sImages').textContent=imgCount;
  document.getElementById('sPairs').textContent=imgCount-1;
  document.getElementById('stats').style.display='flex';
}

function updateCam(){
  camera3d.position.set(
    sph.r*Math.sin(sph.p)*Math.sin(sph.t),
    sph.r*Math.cos(sph.p),
    sph.r*Math.sin(sph.p)*Math.cos(sph.t)
  );
  camera3d.lookAt(pc?pc.position:new THREE.Vector3());
}

function showLoading(msg) {
  document.getElementById('spin').style.display='block';
  document.getElementById('phText').textContent=msg;
  document.getElementById('placeholder').style.display='flex';
}

function hideLoading(msg) {
  document.getElementById('spin').style.display='none';
  document.getElementById('phText').textContent=msg;
}

function setLog(id, msg, type='') {
  document.getElementById(id).innerHTML='<span class="'+type+'">'+msg+'</span>';
}
</script>
</body>
</html>"""


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  SfM 3D Reconstruction — Full System")
    print("  Supports: Images | Video | Live Camera")
    print("  Open browser → http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
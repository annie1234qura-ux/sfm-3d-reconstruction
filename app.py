"""
app.py — Web interface for SfM 3D Reconstruction
=================================================
Install: pip install flask
Run:     python app.py
Open:    http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import cv2
import numpy as np
import os
import glob
import base64
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


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

def run_sfm_on_images(image_paths):
    images = []
    for p in sorted(image_paths):
        img = cv2.imread(p)
        if img is not None:
            # Resize large images for speed
            h, w = img.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)))
            images.append(img)

    if len(images) < 2:
        return [], "Need at least 2 valid images"

    all_points = []
    log = []
    pairs = list(zip(images[:-1], images[1:]))

    for i, (img1, img2) in enumerate(pairs):
        pts1, pts2, good = detect_and_match(img1, img2)
        if pts1 is None or len(good) < 30:
            log.append(f"Pair {i+1}: only {len(good) if good else 0} matches, skipping")
            continue

        K = get_camera_matrix(img1)
        R, t, inliers = estimate_pose(pts1, pts2, K)
        if R is None:
            log.append(f"Pair {i+1}: pose estimation failed")
            continue

        pts3d = triangulate(pts1[inliers], pts2[inliers], K,
                            np.eye(3), np.zeros((3,1)), R, t)
        all_points.append(pts3d)
        log.append(f"Pair {i+1}: {len(good)} matches → {len(pts3d)} 3D points")

    if not all_points:
        return [], "Reconstruction failed. Try more photos with better overlap."

    combined = np.vstack(all_points)
    # Remove outliers
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    mask = np.all(np.abs(combined - mean) < 3 * std, axis=1)
    combined = combined[mask]

    return combined.tolist(), "\n".join(log)


# ─── Routes ─────────────────────────────────────────────────

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SfM 3D Reconstruction</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #080c14;
    --surface: #0d1520;
    --border: #1a2d45;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --text: #c8dff0;
    --dim: #4a6b85;
    --mono: 'Share Tech Mono', monospace;
    --sans: 'Exo 2', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  header {
    text-align: center;
    padding: 3rem 0 2rem;
  }

  header h1 {
    font-family: var(--mono);
    font-size: clamp(1.5rem, 4vw, 2.8rem);
    color: var(--accent);
    letter-spacing: 0.1em;
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
    margin-bottom: 0.5rem;
  }

  header p {
    color: var(--dim);
    font-weight: 300;
    font-size: 1rem;
    letter-spacing: 0.05em;
  }

  .main-grid {
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 1.5rem;
    align-items: start;
  }

  @media (max-width: 800px) {
    .main-grid { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }

  .panel-header {
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--accent);
    letter-spacing: 0.15em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .panel-header::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent);
  }

  .panel-body { padding: 1.2rem; }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: 4px;
    padding: 2rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }

  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent);
    background: rgba(0,212,255,0.05);
  }

  .upload-zone input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    width: 100%;
    height: 100%;
  }

  .upload-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    display: block;
  }

  .upload-zone p {
    font-size: 0.85rem;
    color: var(--dim);
    margin-top: 0.3rem;
  }

  .upload-zone strong {
    color: var(--accent);
    font-family: var(--mono);
  }

  /* Thumbnails */
  #thumbnails {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.4rem;
    margin-top: 1rem;
  }

  #thumbnails img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 2px;
    border: 1px solid var(--border);
    filter: brightness(0.8);
    transition: filter 0.2s;
  }

  #thumbnails img:hover { filter: brightness(1); }

  .file-count {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--dim);
    margin-top: 0.8rem;
    text-align: center;
  }

  /* Button */
  .btn {
    width: 100%;
    padding: 0.9rem;
    margin-top: 1rem;
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    cursor: pointer;
    transition: all 0.2s;
    border-radius: 2px;
    position: relative;
    overflow: hidden;
  }

  .btn::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--accent);
    transform: translateX(-100%);
    transition: transform 0.2s;
    z-index: -1;
  }

  .btn:hover { color: var(--bg); }
  .btn:hover::before { transform: translateX(0); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn:disabled::before { display: none; }
  .btn:disabled:hover { color: var(--accent); }

  /* Log */
  #log {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--dim);
    line-height: 1.8;
    min-height: 80px;
    white-space: pre-wrap;
    margin-top: 1rem;
    padding: 0.8rem;
    background: rgba(0,0,0,0.3);
    border-radius: 2px;
    border-left: 2px solid var(--border);
  }

  #log .ok { color: #00ff88; }
  #log .err { color: var(--accent2); }
  #log .info { color: var(--accent); }

  /* 3D Viewer */
  #viewer {
    width: 100%;
    aspect-ratio: 16/10;
    background: #050810;
    border-radius: 2px;
    position: relative;
    overflow: hidden;
  }

  #viewer canvas { display: block; }

  #viewer-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--dim);
    font-family: var(--mono);
    font-size: 0.8rem;
    gap: 1rem;
    letter-spacing: 0.1em;
  }

  .spinner {
    width: 40px; height: 40px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: none;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .viewer-controls {
    padding: 0.6rem 1.2rem;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.1em;
    display: flex;
    gap: 1.5rem;
    border-top: 1px solid var(--border);
  }

  .stat-bar {
    display: flex;
    gap: 1rem;
    margin-top: 0.8rem;
    flex-wrap: wrap;
  }

  .stat {
    flex: 1;
    min-width: 80px;
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0.6rem;
    text-align: center;
  }

  .stat-val {
    font-family: var(--mono);
    font-size: 1.3rem;
    color: var(--accent);
    display: block;
  }

  .stat-label {
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  /* Progress bar */
  .progress-wrap {
    height: 2px;
    background: var(--border);
    margin-top: 0.8rem;
    border-radius: 1px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: var(--accent);
    width: 0%;
    transition: width 0.3s;
    box-shadow: 0 0 8px var(--accent);
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>◈ STRUCTURE FROM MOTION</h1>
    <p>Upload photos → Get 3D point cloud reconstruction</p>
  </header>

  <div class="main-grid">
    <!-- Left Panel -->
    <div>
      <div class="panel">
        <div class="panel-header">INPUT IMAGES</div>
        <div class="panel-body">
          <div class="upload-zone" id="dropzone">
            <input type="file" id="fileInput" multiple accept="image/*">
            <span class="upload-icon">📁</span>
            <strong>Drop photos here</strong>
            <p>or click to browse</p>
            <p>JPG / PNG • min 5 photos recommended</p>
          </div>
          <div id="thumbnails"></div>
          <div class="file-count" id="fileCount"></div>
          <div class="progress-wrap"><div class="progress-bar" id="progressBar"></div></div>
          <button class="btn" id="runBtn" disabled onclick="runReconstruction()">
            ▶ RUN RECONSTRUCTION
          </button>
          <div id="log">// upload images to begin...</div>
        </div>
      </div>
    </div>

    <!-- Right Panel -->
    <div>
      <div class="panel">
        <div class="panel-header">3D POINT CLOUD VIEWER</div>
        <div id="viewer">
          <div id="viewer-placeholder">
            <div class="spinner" id="spinner"></div>
            <span id="placeholderText">AWAITING RECONSTRUCTION</span>
          </div>
        </div>
        <div class="viewer-controls">
          <span>🖱 LEFT DRAG — rotate</span>
          <span>🖱 RIGHT DRAG — pan</span>
          <span>🖱 SCROLL — zoom</span>
        </div>
      </div>
      <div class="stat-bar" id="statBar" style="display:none">
        <div class="stat"><span class="stat-val" id="statPoints">0</span><span class="stat-label">3D Points</span></div>
        <div class="stat"><span class="stat-val" id="statImages">0</span><span class="stat-label">Images</span></div>
        <div class="stat"><span class="stat-val" id="statPairs">0</span><span class="stat-label">Pairs</span></div>
      </div>
    </div>
  </div>
</div>

<script>
let selectedFiles = [];
let scene, camera, renderer, pointCloud;
let isDragging = false, isRightDrag = false;
let lastMouse = {x:0, y:0};
let spherical = {theta: 0.5, phi: 1.0, radius: 5};

// ── File Handling ──────────────────────────────────────────
const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');

fileInput.addEventListener('change', e => handleFiles(e.target.files));

dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
  selectedFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
  const thumbs = document.getElementById('thumbnails');
  thumbs.innerHTML = '';
  selectedFiles.slice(0, 12).forEach(f => {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    thumbs.appendChild(img);
  });
  document.getElementById('fileCount').textContent =
    `${selectedFiles.length} image${selectedFiles.length !== 1 ? 's' : ''} selected`;
  document.getElementById('runBtn').disabled = selectedFiles.length < 2;
  setLog(`// ${selectedFiles.length} images loaded. Click RUN to reconstruct.`, 'info');
}

// ── Run Reconstruction ─────────────────────────────────────
async function runReconstruction() {
  if (selectedFiles.length < 2) return;

  document.getElementById('runBtn').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('placeholderText').textContent = 'PROCESSING...';
  document.getElementById('progressBar').style.width = '30%';
  setLog('// uploading images...', 'info');

  const form = new FormData();
  selectedFiles.forEach(f => form.append('images', f));

  try {
    document.getElementById('progressBar').style.width = '60%';
    const res = await fetch('/reconstruct', { method: 'POST', body: form });
    const data = await res.json();
    document.getElementById('progressBar').style.width = '100%';

    if (data.error) {
      setLog('ERROR: ' + data.error, 'err');
      document.getElementById('placeholderText').textContent = 'RECONSTRUCTION FAILED';
    } else {
      setLog(data.log + `\n\n✓ Total: ${data.points.length} 3D points reconstructed`, 'ok');
      render3D(data.points);

      // Stats
      document.getElementById('statPoints').textContent = data.points.length;
      document.getElementById('statImages').textContent = selectedFiles.length;
      document.getElementById('statPairs').textContent = selectedFiles.length - 1;
      document.getElementById('statBar').style.display = 'flex';
    }
  } catch(e) {
    setLog('ERROR: ' + e.message, 'err');
  }

  document.getElementById('spinner').style.display = 'none';
  document.getElementById('runBtn').disabled = false;
}

// ── 3D Viewer ──────────────────────────────────────────────
function render3D(points) {
  const container = document.getElementById('viewer');
  document.getElementById('viewer-placeholder').style.display = 'none';

  // Clear old scene
  if (renderer) { renderer.dispose(); container.removeChild(renderer.domElement); }

  const W = container.clientWidth;
  const H = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050810);

  camera = new THREE.PerspectiveCamera(60, W/H, 0.001, 1000);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // Normalize points to center
  const arr = points;
  const xs = arr.map(p=>p[0]), ys = arr.map(p=>p[1]), zs = arr.map(p=>p[2]);
  const cx = (Math.min(...xs)+Math.max(...xs))/2;
  const cy = (Math.min(...ys)+Math.max(...ys))/2;
  const cz = (Math.min(...zs)+Math.max(...zs))/2;
  const maxSpan = Math.max(Math.max(...xs)-Math.min(...xs),
                           Math.max(...ys)-Math.min(...ys),
                           Math.max(...zs)-Math.min(...zs)) || 1;

  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(arr.length * 3);
  const colors = new Float32Array(arr.length * 3);

  arr.forEach((p, i) => {
    positions[i*3]   = (p[0]-cx)/maxSpan;
    positions[i*3+1] = (p[1]-cy)/maxSpan;
    positions[i*3+2] = (p[2]-cz)/maxSpan;

    // Color by height (Y)
    const t = (p[1]-Math.min(...ys))/(Math.max(...ys)-Math.min(...ys)+0.001);
    colors[i*3]   = 0.0 + t * 0.2;
    colors[i*3+1] = 0.6 + t * 0.4;
    colors[i*3+2] = 1.0 - t * 0.3;
  });

  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({ size: 0.025, vertexColors: true, sizeAttenuation: true });
  pointCloud = new THREE.Points(geo, mat);
  scene.add(pointCloud);

  // Grid
  const grid = new THREE.GridHelper(2, 20, 0x1a2d45, 0x0d1a28);
  grid.position.y = -0.6;
  scene.add(grid);

  spherical = { theta: 0.5, phi: 1.0, radius: 2.5 };
  updateCamera();

  // Mouse controls
  renderer.domElement.addEventListener('mousedown', e => {
    isDragging = true;
    isRightDrag = e.button === 2;
    lastMouse = { x: e.clientX, y: e.clientY };
  });
  renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
  window.addEventListener('mouseup', () => isDragging = false);
  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    if (!isRightDrag) {
      spherical.theta -= dx * 0.008;
      spherical.phi = Math.max(0.1, Math.min(Math.PI-0.1, spherical.phi + dy * 0.008));
    } else {
      const panSpeed = 0.003 * spherical.radius;
      pointCloud.position.x += dx * panSpeed;
      pointCloud.position.y -= dy * panSpeed;
    }
    lastMouse = { x: e.clientX, y: e.clientY };
    updateCamera();
  });
  renderer.domElement.addEventListener('wheel', e => {
    spherical.radius = Math.max(0.5, Math.min(20, spherical.radius + e.deltaY * 0.005));
    updateCamera();
  });

  function animate() {
    requestAnimationFrame(animate);
    pointCloud.rotation.y += 0.001;
    renderer.render(scene, camera);
  }
  animate();
}

function updateCamera() {
  camera.position.set(
    spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta),
    spherical.radius * Math.cos(spherical.phi),
    spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta)
  );
  camera.lookAt(pointCloud ? pointCloud.position : new THREE.Vector3());
}

function setLog(msg, type='') {
  const el = document.getElementById('log');
  el.innerHTML = `<span class="${type}">${msg}</span>`;
}
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({'error': 'Need at least 2 images'})

    # Save uploaded files
    paths = []
    for f in files:
        fname = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, fname)
        f.save(path)
        paths.append(path)

    # Run SfM
    points, log = run_sfm_on_images(paths)

    # Cleanup
    for p in paths:
        try: os.remove(p)
        except: pass

    if not points:
        return jsonify({'error': log})

    return jsonify({'points': points, 'log': log})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  SfM Web App running!")
    print("  Open your browser → http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
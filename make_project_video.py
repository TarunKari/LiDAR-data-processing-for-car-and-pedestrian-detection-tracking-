"""
make_project_video.py
=====================
Generates a project demo video from LiDAR pipeline frame images.

Usage:
    python make_project_video.py

Requirements:
    pip install opencv-python pillow numpy

Folder structure expected (run main.py --visualize first):
    output/
    └── frames/
        ├── frame_2415_overview.png
        ├── frame_2415_pedestrian.png
        ├── frame_2415_car_cyclist.png
        ├── frame_2415_topdown.png
        ├── frame_2416_overview.png
        └── ...
"""

import os
import sys
import glob

# ── Check dependencies first ─────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed.")
    print("Fix:   pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed.")
    print("Fix:   pip install numpy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: pillow not installed.")
    print("Fix:   pip install pillow")
    sys.exit(1)

# ── Paths — relative to wherever you run this script from ────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(SCRIPT_DIR, 'output', 'frames')
OUT_PATH   = os.path.join(SCRIPT_DIR, 'output', 'project_demo.mp4')

# ── Check frames folder exists ───────────────────────────────────────────────
if not os.path.exists(FRAMES_DIR):
    print(f"ERROR: Frames folder not found: {FRAMES_DIR}")
    print("Fix:   Run  python main.py --data_dir data/ --output_dir output/ --visualize")
    sys.exit(1)

all_pngs = sorted(glob.glob(os.path.join(FRAMES_DIR, '*.png')))
if len(all_pngs) == 0:
    print(f"ERROR: No PNG images found in {FRAMES_DIR}")
    print("Fix:   Run  python main.py --data_dir data/ --output_dir output/ --visualize")
    sys.exit(1)

print(f"Found {len(all_pngs)} frame images in: {FRAMES_DIR}")

# ── Settings ─────────────────────────────────────────────────────────────────
W, H = 1920, 1080
FPS  = 30

# Colours (BGR for OpenCV)
BG    = (17,  17,  17)
CYAN  = (230, 216,  90)   # BGR of #5ad8e6
WHITE = (255, 255, 255)
GREY  = (160, 160, 160)
GREEN = (80,  200,  80)

# ── Helper functions ─────────────────────────────────────────────────────────
def blank():
    return np.full((H, W, 3), BG, dtype=np.uint8)

def put(frame, text, x, y, size=0.8, color=WHITE, thickness=2, bold=False):
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (x, y), font, size, color, thickness, cv2.LINE_AA)

def put_center(frame, text, y, size=0.8, color=WHITE, thickness=2, bold=False):
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, size, thickness)
    put(frame, text, (W - tw) // 2, y, size, color, thickness, bold)

def fit_image(img_path, tw, th):
    """Load image and fit inside target size, centred on dark background."""
    img = Image.open(img_path).convert('RGB')
    iw, ih = img.size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('RGB', (tw, th), (17, 17, 17))
    canvas.paste(img, ((tw - nw) // 2, (th - nh) // 2))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

def crossfade(f1, f2, steps=15):
    for i in range(steps):
        a = i / steps
        yield cv2.addWeighted(f1, 1 - a, f2, a, 0)

def write_n(vw, frame, n):
    for _ in range(n):
        vw.write(frame)

# ── Build ordered frame list ──────────────────────────────────────────────────
# Detect frame IDs from filenames
frame_ids = sorted(set(
    os.path.basename(f).split('_')[1]
    for f in all_pngs
    if os.path.basename(f).startswith('frame_')
))

view_order  = ['overview', 'pedestrian', 'car_cyclist', 'topdown']
view_labels = ['3D Overview', 'Pedestrian Focus', 'Car & Cyclist Focus', 'Top-Down View']

print(f"Frame IDs detected: {frame_ids}")
print(f"Output video:       {OUT_PATH}")
print("Building video...")

# ── Video writer ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

if not vw.isOpened():
    print("ERROR: Could not open VideoWriter. Check ffmpeg/opencv installation.")
    sys.exit(1)

# ── SECTION 1: Title card ─────────────────────────────────────────────────────
print("  [1/5] Title card...")
title = blank()
cv2.rectangle(title, (0, 0),    (W, 6), CYAN, -1)
cv2.rectangle(title, (0, H-6), (W, H),  CYAN, -1)
put_center(title, 'LiDAR-Based Object Detection', 340, 1.8, WHITE, 3, bold=True)
put_center(title, 'and Tracking Pipeline',         415, 1.8, WHITE, 3, bold=True)
put_center(title, 'Blickfeld Cube 1  |  CCR = 1.0000  |  FAR = 0.0000 /hr', 510, 0.75, CYAN, 2)
put_center(title, 'IU International University of Applied Sciences', 575, 0.65, GREY, 1)

black = blank()
for f in crossfade(black, title, 20):
    vw.write(f)
write_n(vw, title, 70)
prev = title

# ── SECTION 2: Per-frame views ────────────────────────────────────────────────
print("  [2/5] Frame views...")
TOP_BAR = 60
BOT_BAR = 50
IMG_H   = H - TOP_BAR - BOT_BAR

for fid in frame_ids:
    for view, vlabel in zip(view_order, view_labels):
        img_path = os.path.join(FRAMES_DIR, f'frame_{fid}_{view}.png')
        if not os.path.exists(img_path):
            continue

        canvas = blank()

        # Image area
        img_cv = fit_image(img_path, W, IMG_H)
        canvas[TOP_BAR:TOP_BAR+IMG_H, :] = img_cv

        # Top HUD bar
        cv2.rectangle(canvas, (0, 0), (W, TOP_BAR), (17,17,17), -1)
        cv2.rectangle(canvas, (0, TOP_BAR), (W, TOP_BAR+3), CYAN, -1)
        put(canvas, f'Frame {fid}', 15, 40, 0.85, CYAN, 2, bold=True)
        put(canvas, vlabel,         160, 40, 0.85, WHITE, 2)
        put(canvas, 'CCR=1.0000  FAR=0.0000/hr', W-360, 40, 0.65, GREY, 1)

        # Bottom bar
        cv2.rectangle(canvas, (0, H-BOT_BAR), (W, H), (17,17,17), -1)
        cv2.rectangle(canvas, (0, H-BOT_BAR), (W, H-BOT_BAR+3), CYAN, -1)
        put(canvas, 'LiDAR Detection & Tracking  |  Blickfeld Cube 1  |  IU Applied Sciences', 15, H-15, 0.52, GREY, 1)

        # Frame progress dots
        for i, f_id in enumerate(frame_ids):
            cx = W - 120 + i * 28
            col = CYAN if f_id == fid else (70, 70, 70)
            cv2.circle(canvas, (cx, H-25), 7, col, -1)

        for f in crossfade(prev, canvas, 12):
            vw.write(f)
        write_n(vw, canvas, 45)
        prev = canvas

# ── SECTION 3: 2×2 comparison grid ───────────────────────────────────────────
print("  [3/5] Comparison grid...")
comp = blank()
cv2.rectangle(comp, (0, 0),   (W, 6),   CYAN, -1)
cv2.rectangle(comp, (0, H-6),(W, H),    CYAN, -1)
put_center(comp, 'All Frames - Overview Comparison', 45, 0.9, WHITE, 2, bold=True)

pad = 12
cw  = (W - 3*pad) // 2
ch  = (H - 60 - 3*pad) // 2
positions = [
    (pad,       60+pad),
    (2*pad+cw,  60+pad),
    (pad,       60+2*pad+ch),
    (2*pad+cw,  60+2*pad+ch),
]
for i, fid in enumerate(frame_ids[:4]):
    x, y = positions[i]
    p = os.path.join(FRAMES_DIR, f'frame_{fid}_overview.png')
    if os.path.exists(p):
        comp[y:y+ch, x:x+cw] = fit_image(p, cw, ch)
    cv2.rectangle(comp, (x, y), (x+cw, y+ch), CYAN, 2)
    put(comp, f'Frame {fid}', x+10, y+30, 0.7, CYAN, 2, bold=True)

for f in crossfade(prev, comp, 15):
    vw.write(f)
write_n(vw, comp, 80)
prev = comp

# ── SECTION 4: Performance summary ───────────────────────────────────────────
print("  [4/5] Performance summary...")
perf = blank()
cv2.rectangle(perf, (0, 0),   (W, 6),  CYAN, -1)
cv2.rectangle(perf, (0, H-6),(W, H),   CYAN, -1)
put_center(perf, 'Performance Results vs. Project Targets', 80, 1.0, WHITE, 2, bold=True)

results = [
    ('CCR',          '1.0000', '>= 0.99'),
    ('FAR (/ hr)',   '0.0000', '<= 0.01'),
    ('GT Objects',   '3,268',  '100% correct'),
    ('False Alarms', '0',      'Zero errors'),
    ('Track Confirm','98.3%',  '2,205 / 2,242'),
]
rw, rh  = 310, 110
gap_r   = 22
total_r = len(results)*rw + (len(results)-1)*gap_r
rx0     = (W - total_r) // 2
ry      = 320

for i, (lbl, val, sub) in enumerate(results):
    x = rx0 + i*(rw+gap_r)
    cv2.rectangle(perf, (x, ry),        (x+rw, ry+rh),    (20,60,20), -1)
    cv2.rectangle(perf, (x, ry),        (x+rw, ry+rh),    GREEN, 2)
    # value
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw,th),_ = cv2.getTextSize(val, font, 1.0, 2)
    cv2.putText(perf, val, (x+(rw-tw)//2, ry+45), font, 1.0, GREEN, 2, cv2.LINE_AA)
    # label
    (tw2,_),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(perf, lbl, (x+(rw-tw2)//2, ry+75),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    (tw3,_),_ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(perf, sub, (x+(rw-tw3)//2, ry+100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)

for f in crossfade(prev, perf, 15):
    vw.write(f)
write_n(vw, perf, 85)
prev = perf

# ── SECTION 5: End card ───────────────────────────────────────────────────────
print("  [5/5] End card...")
end = blank()
cv2.rectangle(end, (0, 0),   (W, 6), CYAN, -1)
cv2.rectangle(end, (0, H-6),(W, H),  CYAN, -1)
put_center(end, 'LiDAR Object Detection & Tracking', 380, 1.5, WHITE, 3, bold=True)
put_center(end, 'CCR = 1.0000  |  FAR = 0.0000 / hr  |  Targets Met', 460, 0.85, CYAN, 2)
put_center(end, 'IU International University of Applied Sciences  |  Blickfeld Cube 1', 540, 0.65, GREY, 1)

for f in crossfade(prev, end, 20):
    vw.write(f)
write_n(vw, end, 80)

# Fade out
for i in range(25):
    vw.write(cv2.addWeighted(end, 1 - i/25, black, i/25, 0))

vw.release()

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"\n✓ Video saved: {OUT_PATH}")
print(f"  Size: {size_mb:.1f} MB")
print(f"  Resolution: {W}x{H}  |  FPS: {FPS}")
print("\nDone!")

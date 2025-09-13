#!/usr/bin/env python3
import cv2, subprocess, re, os, numpy as np, math, sys

try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False

def is_rgb_device(dev):
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
            capture_output=True, text=True, timeout=2
        ).stdout
        return bool(re.search(r"(MJPG|YUYV|RGB3)", out))
    except:
        return False

def list_rgb_devices():
    nodes = [f"/dev/{d}" for d in os.listdir("/dev") if d.startswith("video")]
    return [dev for dev in sorted(nodes) if is_rgb_device(dev)]

def realsense_serials():
    serials = {}
    if not HAS_RS:
        return serials
    ctx = rs.context()
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        serials[serial] = name
    return serials

def make_grid(frames, w=320, h=240):
    n = len(frames)
    if n == 0:
        return None
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    frames_resized = [cv2.resize(f, (w, h)) for f in frames]
    blank = np.zeros_like(frames_resized[0])
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            row.append(frames_resized[idx] if idx < n else blank)
        grid.append(np.hstack(row))
    return np.vstack(grid)

def capture_all():
    rgb_nodes = list_rgb_devices()
    rs_info = realsense_serials()
    caps = []

    for dev in rgb_nodes:
        idx = int(dev.replace("/dev/video", ""))
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            label = f"{dev} (idx {idx})"
            # If it's a RealSense, attach serial or name
            if HAS_RS and "RealSense" in cap.getBackendName():
                # pyrealsense2 doesn’t map to /dev/video*, so just list all
                if rs_info:
                    serial_list = ", ".join([f"{s} ({n})" for s, n in rs_info.items()])
                    label += f" [RealSense {serial_list}]"
                else:
                    label += " [RealSense]"
            caps.append((dev, idx, cap, label))
            print(f"Camera {label}")
        else:
            print(f"Could not open {dev}")

    if not caps:
        print("No RGB cameras found.")
        return

    try:
        while True:
            frames = []
            for dev, idx, cap, label in caps:
                ret, frame = cap.read()
                if ret:
                    cv2.putText(frame, label, (5,20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    frames.append(frame)
            grid = make_grid(frames)
            if grid is not None:
                cv2.imshow("Cameras", grid)
            key = cv2.waitKey(1)
            if key in (27, ord('q')):
                break
    finally:
        for _, _, cap, _ in caps: cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        capture_all()
    except KeyboardInterrupt:
        sys.exit(0)

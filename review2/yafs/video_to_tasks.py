# video_to_tasks.py
# Usage:
#  python video_to_tasks.py --video videos/road.mp4 --out tasks.csv --frame-skip 5 --node-id edge-1

import cv2, csv, argparse, os
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to input video file")
parser.add_argument("--out", default="tasks.csv", help="Output CSV file")
parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame")
parser.add_argument("--node-id", default="edge-1", help="Origin node id (client)")
parser.add_argument("--min-area", type=int, default=800, help="Min contour area to count")
parser.add_argument("--base-work", type=float, default=500.0, help="Base MI assigned per detection")
parser.add_argument("--high-area", type=int, default=5000, help="Area above which mark HIGH priority")
parser.add_argument("--deadline-low", type=int, default=5000, help="deadline ms for low priority")
parser.add_argument("--deadline-high", type=int, default=2000, help="deadline ms for high priority")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit("Cannot open video: " + args.video)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
bgsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

out_rows = []
task_id = 0
frame_idx = 0

print("Processing video -> creating tasks CSV ...")
pbar = tqdm(total=frame_count//max(1,args.frame_skip)+1)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % args.frame_skip != 0:
        continue

    timestamp = frame_idx / fps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg = bgsub.apply(gray)
    # morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < args.min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        bbox_area = w*h
        # map area to work MI (simple linear scaling)
        work_mi = args.base_work * max(1.0, bbox_area / 1000.0)
        # approximate size in bytes for network transfer (just an estimate)
        size_bytes = int(bbox_area * 50)  # tune factor as needed
        priority = "high" if bbox_area >= args.high_area else "low"
        deadline_ms = args.deadline_high if priority=="high" else args.deadline_low

        task = {
            "timestamp_s": round(timestamp, 3),
            "node_id": args.node_id,
            "task_id": f"t{task_id}",
            "frame_idx": frame_idx,
            "work_mi": round(work_mi,2),
            "size_bytes": size_bytes,
            "priority": priority,
            "deadline_ms": int(deadline_ms)
        }
        out_rows.append(task)
        task_id += 1

    pbar.update(1)

pbar.close()
cap.release()

# write CSV
fieldnames = ["timestamp_s","node_id","task_id","frame_idx","work_mi","size_bytes","priority","deadline_ms"]
with open(args.out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in out_rows:
        writer.writerow(r)

print(f"Wrote {len(out_rows)} tasks to {args.out}")

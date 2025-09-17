import cv2
import time
import yaml
from yolo_detector import YoloDetector
from foreground_gfm import ForegroundGFM
from congestion_cnn import CongestionJudge
from emergency_detect import EmergencyVehicleDetector
from speedometer_ci import Speedometer
from traffic_flow import TrafficFlowEstimator
from scheduler import Scheduler

# Load configuration
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Initialize components
video = cv2.VideoCapture("data/traffic_video.mp4")  # Video file input
yolo = YoloDetector(cfg)
gfm = ForegroundGFM(cfg)
cong = CongestionJudge(cfg)
emer = EmergencyVehicleDetector(cfg)
spd = Speedometer(cfg)
flow = TrafficFlowEstimator(cfg)
scheduler = Scheduler(cfg)

frame_id = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Step 1: Vehicle Detection
    dets = yolo.detect(frame, run=(frame_id % cfg["frame_skip_yolo"] == 0))

    # Step 2: Foreground Detection
    if scheduler.decide("foreground") == "local":
        fgmask = gfm.mask(frame)

    # Step 3: Congestion Detection
    congested, coverage = False, 0
    if scheduler.decide("congestion") == "local":
        congested, coverage = cong.evaluate(frame, fgmask, dets)

    # Step 4: Emergency Vehicle Detection
    emergency, ebox = False, None
    if scheduler.decide("emergency") == "local":
        emergency, ebox = emer.detect(frame, dets)

    # Step 5: Speed Estimation (CI-based MLP)
    speeds = []
    if scheduler.decide("speed") == "local":
        speeds = spd.update(frame, dets)  # Uses MLP model for adaptive speed estimation

    # Step 6: Traffic Flow Estimation
    count = 0
    if scheduler.decide("count") == "local":
        count = flow.update(frame, dets)

    # Display Status on Frame
    speed_text = f"Speeds: {', '.join([f'{int(s)} km/h' for s in speeds])}" if speeds else "Speeds: --"
    status = f"Congestion: {congested}, Emergency: {emergency}, Flow Count: {count}"
    cv2.putText(frame, speed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Smart Traffic Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

video.release()
cv2.destroyAllWindows()

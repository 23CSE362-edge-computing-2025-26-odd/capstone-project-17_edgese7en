import cv2
import numpy as np

class YoloDetector:
    def _init_(self, config):
        # Load YOLO model (Tiny version for Edge)
        self.net = cv2.dnn.readNetFromDarknet(
            config["yolo_config_path"], 
            config["yolo_weights_path"]
        )
        self.classes = open(config["coco_names_path"]).read().splitlines()

    def detect(self, frame, run=True):
        if not run:
            return []

        # Prepare input blob for YOLOv3-Tiny
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1/255, size=(320, 320), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # Get network output
        output_layers = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(output_layers)

        height, width = frame.shape[:2]
        detections = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only detect vehicles (car, bus, truck, motorbike)
                if confidence > 0.4 and self.classes[class_id] in ["car", "bus", "truck", "motorbike"]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    detections.append({
                        "class": self.classes[class_id],
                        "box": [x, y, w, h],
                        "center": [center_x, center_y],
                        "confidence": float(confidence)
                    })

        return detections
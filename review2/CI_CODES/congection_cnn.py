import cv2
import numpy as np
import time
import tensorflow as tf

class CongestionJudge:
    def _init_(self, cfg):
        self.roi = cfg["congestion_roi"]
        self.area_threshold = cfg["congestion_area_threshold"]
        self.duration_threshold = cfg["congestion_duration_threshold"]
        self.congestion_duration = 0
        self.last_time = time.time()

        # Load the pre-trained TFLite model
        self.interpreter = tf.lite.Interpreter(model_path="models/congestion_cnn.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)

    def evaluate(self, frame, fgmask_gfm, detections):
        x, y, w, h = self.roi
        roi_gfm = fgmask_gfm[y:y+h, x:x+w]
        roi_ziv = self.bg_subtractor.apply(frame)[y:y+h, x:x+w]

        congestion_area = np.sum((roi_gfm == 255) & (roi_ziv == 0))
        current_time = time.time()

        if congestion_area > self.area_threshold:
            self.congestion_duration += (current_time - self.last_time)
        else:
            self.congestion_duration = max(0, self.congestion_duration - (current_time - self.last_time))

        self.last_time = current_time

        if self.congestion_duration > self.duration_threshold:
            # Pass ROI image to CNN model
            roi_img = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_img, (224, 224))
            input_data = np.expand_dims(roi_resized.astype(np.float32) / 255.0, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

            if pred >= 0.5:
                return True, congestion_area  # Confirmed Congestion

        return False, congestion_area

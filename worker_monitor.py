import cv2
import numpy as np
import os
import logging
import json
import urllib.request
import time

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                   handlers=[logging.FileHandler("activity_log.txt"), logging.StreamHandler()])

class WorkerActivityMonitor:
    def __init__(self, config_path="config.json"):
        # Load config or use defaults
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {
                "video_source": 0, "roi_x": 0, "roi_y": 0, "roi_width": 640, "roi_height": 480,
                "resize_width": 640, "resize_height": 480, "working_frames_threshold": 10, 
                "idle_frames_threshold": 30, "show_video": True, "confidence_threshold": 0.5
            }
        
        # Initialize state
        self.is_working = False
        self.consecutive_active_frames = 0
        self.consecutive_inactive_frames = 0
        
        # Setup model
        self._setup_model()
    
    def _setup_model(self):
        """Download and initialize the model"""
        # Model file paths
        model_path = self.config.get("model_path", "models/MobileNetSSD_deploy.caffemodel")
        config_path = self.config.get("model_config", "models/MobileNetSSD_deploy.prototxt")
        os.makedirs("models", exist_ok=True)
        
        # Download model files if needed
        for file_path, url in [
            (model_path, "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"),
            (config_path, "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt")
        ]:
            if not os.path.exists(file_path):
                logging.info(f"Downloading {file_path}...")
                urllib.request.urlretrieve(url, file_path)
        
        # Initialize model
        self.model = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                      "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        logging.info("Model ready")
            
    def detect_activity(self, frame):
        """Detect if worker is present and active"""
        # Resize and get ROI
        frame = cv2.resize(frame, (self.config["resize_width"], self.config["resize_height"]))
        roi = frame[self.config["roi_y"]:self.config["roi_y"]+self.config["roi_height"], 
                   self.config["roi_x"]:self.config["roi_x"]+self.config["roi_width"]]
        
        # Run person detection
        h, w = roi.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
        self.model.setInput(blob)
        detections = self.model.forward()
        
        # Check for persons
        person_count = 0
        person_boxes = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config.get("confidence_threshold", 0.5):
                idx = int(detections[0, 0, i, 1])
                if self.classes[idx] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    person_boxes.append(box.astype("int"))
                    person_count += 1
        
        # Update activity counters
        is_active = person_count > 0
        if is_active:
            self.consecutive_active_frames += 1
            self.consecutive_inactive_frames = 0
        else:
            self.consecutive_inactive_frames += 1
            self.consecutive_active_frames = 0
        
        # Update working status
        prev_status = self.is_working
        self.is_working = (self.consecutive_active_frames >= self.config["working_frames_threshold"]) or \
                         (self.is_working and self.consecutive_inactive_frames < self.config["idle_frames_threshold"])
        
        # Log status changes
        if self.is_working != prev_status:
            logging.info(f"Worker status: {'Working' if self.is_working else 'Idle'}")
        
        # Visualization
        if self.config["show_video"]:
            # Draw ROI
            cv2.rectangle(frame, (self.config["roi_x"], self.config["roi_y"]),
                        (self.config["roi_x"] + self.config["roi_width"], 
                         self.config["roi_y"] + self.config["roi_height"]), (0, 255, 0), 2)
            
            # Draw people boxes
            for box in person_boxes:
                startX, startY, endX, endY = box
                cv2.rectangle(frame, 
                            (startX + self.config["roi_x"], startY + self.config["roi_y"]),
                            (endX + self.config["roi_x"], endY + self.config["roi_y"]), 
                            (0, 0, 255), 2)
            
            # Add status text
            cv2.putText(frame, f"Status: {'Working' if self.is_working else 'Idle'}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"People: {person_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Worker Monitoring", frame)
        
        return self.is_working
        
    def run(self):
        """Main monitoring loop"""
        # Initialize video capture
        video_source = self.config["video_source"]
        cap = cv2.VideoCapture(int(video_source) if isinstance(video_source, int) or not os.path.exists(video_source) else video_source)
        
        if not cap.isOpened():
            logging.error(f"Cannot open video source: {video_source}")
            return
            
        logging.info("Monitoring started")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # For video files, loop back to beginning
                    if isinstance(video_source, str) and os.path.exists(video_source):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                self.detect_activity(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Monitoring stopped")

# Run the application
if __name__ == "__main__":
    WorkerActivityMonitor().run()

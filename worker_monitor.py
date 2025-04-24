import cv2
import numpy as np
import time
import os
import logging
import json
import urllib.request
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("activity_log.txt"),
        logging.StreamHandler()
    ]
)

class WorkerActivityMonitor:
    def __init__(self, config_path="config.json"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize variables
        self.last_activity_time = time.time()
        self.is_working = False
        self.frame_count = 0
        self.consecutive_active_frames = 0
        self.consecutive_inactive_frames = 0
        
        # Initialize the human detection model
        self._ensure_model_exists()
        self._init_model()
    
    def _ensure_model_exists(self):
        """Make sure model files exist, download if needed"""
        model_path = self.config.get("model_path", "models/MobileNetSSD_deploy.caffemodel")
        config_path = self.config.get("model_config", "models/MobileNetSSD_deploy.prototxt")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Download model file if it doesn't exist
        if not os.path.exists(model_path):
            logging.info(f"Downloading model file to {model_path}...")
            try:
                url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
                urllib.request.urlretrieve(url, model_path)
                logging.info("Model file downloaded successfully.")
            except Exception as e:
                logging.error(f"Failed to download model file: {e}")
                raise
                
        # Download config file if it doesn't exist
        if not os.path.exists(config_path):
            logging.info(f"Downloading model config to {config_path}...")
            try:
                url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"
                urllib.request.urlretrieve(url, config_path)
                logging.info("Model config downloaded successfully.")
            except Exception as e:
                logging.error(f"Failed to download model config: {e}")
                raise

    def _init_model(self):
        """Initialize the lightweight detection model"""
        try:
            # We'll use OpenCV's DNN module with a pre-trained MobileNet SSD model
            model_path = self.config.get("model_path", "models/MobileNetSSD_deploy.caffemodel")
            config_path = self.config.get("model_config", "models/MobileNetSSD_deploy.prototxt")
            
            self.model = cv2.dnn.readNetFromCaffe(config_path, model_path)
            
            # Use OpenCV's DNN optimization (can be CPU or GPU based on hardware)
            if self.config.get("use_gpu", False) and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
            # MobileNet SSD was trained on these classes
            self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                           "sofa", "train", "tvmonitor"]
            
            logging.info("Human detection model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
            
    def _detect_humans(self, frame):
        """Detect humans in the frame using the DNN model"""
        # Get frame dimensions
        (h, w) = frame.shape[:2]
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, 
                                     (300, 300), 127.5)
        
        # Pass the blob through the network
        self.model.setInput(blob)
        detections = self.model.forward()
        
        # Initialize the count of people and activity level
        person_count = 0
        person_boxes = []
        activity_level = 0
        
        # Loop over the detections
        for i in range(detections.shape[2]):
            # Extract the confidence
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > self.config.get("confidence_threshold", 0.5):
                # Extract the index of the class label
                idx = int(detections[0, 0, i, 1])
                
                # Check if the detection is a person
                if self.classes[idx] == "person":
                    person_count += 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # Calculate area of detection (used as a proxy for activity level)
                    area = (endX - startX) * (endY - startY)
                    activity_level += area
                    person_boxes.append((startX, startY, endX, endY))
        
        return person_count, person_boxes, activity_level
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            # Return default config
            return {
                "video_source": 0,  # Default to webcam
                "roi_x": 0,
                "roi_y": 0,
                "roi_width": 640,
                "roi_height": 480,
                "resize_width": 640,
                "resize_height": 480,
                "activity_threshold": 5000,
                "working_frames_threshold": 10,
                "idle_frames_threshold": 30,
                "bg_history": 500,
                "bg_threshold": 16,
                "show_video": True,
                "model_path": "models/MobileNetSSD_deploy.caffemodel",
                "model_config": "models/MobileNetSSD_deploy.prototxt",
                "confidence_threshold": 0.5,
                "use_gpu": False
            }
            
    def _extract_roi(self, frame):
        """Extract region of interest from frame"""
        x, y = self.config["roi_x"], self.config["roi_y"]
        w, h = self.config["roi_width"], self.config["roi_height"]
        
        # Make sure ROI is within frame bounds
        h_frame, w_frame = frame.shape[:2]
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        return frame[y:y+h, x:x+w]
    
    def process_frame(self, frame):
        """Process a single frame to detect worker activity"""
        # Resize frame for faster processing
        frame = cv2.resize(frame, (self.config["resize_width"], self.config["resize_height"]))
        
        # Extract ROI
        roi = self._extract_roi(frame)
        
        # Detect humans in the frame using only the model
        person_count, person_boxes, activity_level = self._detect_humans(roi)
        
        # Consider active if a person is detected
        is_active = person_count > 0
        
        # Update consecutive frame counters
        if is_active:
            self.consecutive_active_frames += 1
            self.consecutive_inactive_frames = 0
        else:
            self.consecutive_inactive_frames += 1
            self.consecutive_active_frames = 0
        
        # Update working status
        previous_status = self.is_working
        
        if self.consecutive_active_frames >= self.config["working_frames_threshold"]:
            self.is_working = True
        elif self.consecutive_inactive_frames >= self.config["idle_frames_threshold"]:
            self.is_working = False
            
        # Log status change
        if self.is_working != previous_status:
            status_text = "Working" if self.is_working else "Idle"
            logging.info(f"Worker status changed to: {status_text}")
            
        # Prepare visualization
        if self.config["show_video"]:
            # Draw ROI
            cv2.rectangle(frame, 
                         (self.config["roi_x"], self.config["roi_y"]),
                         (self.config["roi_x"] + self.config["roi_width"], 
                          self.config["roi_y"] + self.config["roi_height"]),
                         (0, 255, 0), 2)
            
            # Draw person boxes
            for (startX, startY, endX, endY) in person_boxes:
                # Adjust coordinates to account for ROI offset
                startX += self.config["roi_x"]
                startY += self.config["roi_y"]
                endX += self.config["roi_x"]
                endY += self.config["roi_y"]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            
            # Add status text
            status_text = f"Status: {'Working' if self.is_working else 'Idle'}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add person count and activity level
            cv2.putText(frame, f"People: {person_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Activity: {activity_level}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("Worker Monitoring", frame)
        
        return frame, self.is_working
        
    def run(self):
        """Main loop to process video"""
        # Initialize video capture
        video_source = self.config["video_source"]
        if isinstance(video_source, str) and os.path.exists(video_source):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(int(video_source))
            
        if not cap.isOpened():
            logging.error(f"Error opening video source: {video_source}")
            return
            
        logging.info("Worker monitoring started")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    # If using a file, restart when it ends
                    if isinstance(video_source, str) and os.path.exists(video_source):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Process the frame
                processed_frame, is_working = self.process_frame(frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logging.error(f"Error in worker monitoring: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Worker monitoring stopped")

if __name__ == "__main__":
    monitor = WorkerActivityMonitor()
    monitor.run()

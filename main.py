import cv2
import time
import logging
from activity_detector import WorkerActivityDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    cap = cv2.VideoCapture(0)  # Change to 'path/to/video.mp4' for testing with a file
    if not cap.isOpened():
        logging.error("Failed to open video stream.")
        return

    detector = WorkerActivityDetector(model_path="model/worker_activity_model.onnx")

    logging.info("Starting Worker Activity Monitor...")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Unable to read frame.")
            break

        is_working = detector.detect(frame)
        status_text = "Working" if is_working else "Idle"
        logging.info(f"Worker Activity: {status_text}")

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Worker Activity Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

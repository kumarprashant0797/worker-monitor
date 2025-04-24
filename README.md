# Worker Activity Monitoring System

This project is a standalone, vision-based system that detects whether a worker is actively engaged in a task or idle. It is optimized for low-cost hardware and automatically starts on system boot.

---

## Features

- Real-time video capture from webcam or video file
- Lightweight ONNX-based inference for activity detection
- Logs and displays worker activity status ("Working" or "Idle")
- Auto-starts on boot using systemd

---

## Project Structure

```
worker-monitor/
├── main.py                      # Entry point for video stream capture and inference
├── activity_detector.py         # Preprocessing and ONNX model inference logic
├── requirements.txt             # Required Python packages
├── worker-monitor.service       # systemd service file for auto-start
└── model/
    └── worker_activity_model.onnx  # Placeholder ONNX model
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download or Replace Model

Replace `model/worker_activity_model.onnx` with a trained ONNX model that outputs a probability score for "Working" classification.

> You can train your own using PyTorch/TensorFlow and export it to ONNX, or download one from ONNX Model Zoo.

### 3. Run the Application

```bash
python main.py
```

> Modify the video source in `main.py` if you want to test on a saved video file instead of webcam.

### 4. Setup Auto-Start with Systemd (Linux only)

1. Copy `worker-monitor.service` to `/etc/systemd/system/`:
   ```bash
   sudo cp worker-monitor.service /etc/systemd/system/
   ```

2. Reload services:
   ```bash
   sudo systemctl daemon-reload
   ```

3. Enable and start service:
   ```bash
   sudo systemctl enable worker-monitor
   sudo systemctl start worker-monitor
   ```

4. Check logs:
   ```bash
   sudo journalctl -u worker-monitor.service
   ```

---

## Commit Messages

Here are suggested Git commit messages for each file:

- `main.py`: `feat: add main app for real-time worker activity monitoring using webcam`
- `activity_detector.py`: `feat: implement ONNX-based worker activity detection module`
- `requirements.txt`: `chore: add dependency list for OpenCV, NumPy, and ONNX runtime`
- `worker-monitor.service`: `feat: add systemd service file for auto-start on boot`
- `model/worker_activity_model.onnx`: `chore: add placeholder ONNX model file`
- `README.md`: `docs: add full project README with setup instructions and usage`

---

## License

MIT License

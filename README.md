# Real-Time Object Detection with YOLOv5

This project uses a YOLOv5 object detection model to detect objects in real-time on a video stream from the wrist mounted camera. The project includes:
- **Asynchronous Video Capture** for optimized video processing with OpenCV.
- **YOLOv5 Object Detection** to detect and label objects in each frame.
- **Kinova Gen3 Robot API** integration for session management and communication setup (via TCP/IP).

## Requirements

1. **Python 3.7+**
2. **Dependencies**
     - `torch`
     - `opencv-python`
     - `numpy`
   - Additional dependencies for YOLOv5:
     ```bash
     pip install ultralytics
     ```

## Usage

### 1. Project Structure

- `camera.py`: Main object detection script using YOLOv5.
- `videocaptureasync.py`: A helper class for asynchronous video capture.

### 2. Setup

Ensure you have access to an IP camera with the RTSP link.

1. **Edit IP Stream Source**  
   Replace the `stream` variable in `camera.py` with your IP camera's RTSP URL:
   ```python
   test = ObjectDetetction("rtsp://<your-camera-ip>/color")
2. Run the main file
   ```python
   camera.py
   

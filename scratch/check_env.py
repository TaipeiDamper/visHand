
import mediapipe as mp
import cv2
print(f"MediaPipe version: {mp.__version__}")
print(f"OpenCV version: {cv2.__version__}")
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("MediaPipe Tasks API is available.")
except ImportError:
    print("MediaPipe Tasks API is NOT available.")

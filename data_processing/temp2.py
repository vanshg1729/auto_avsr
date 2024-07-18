import os
import sys

print(f"temp2.py: {__file__}")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preparation.detectors.yoloface.face_detector import YoloDetector

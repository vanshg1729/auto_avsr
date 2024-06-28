import time
import cv2
import torchvision.io as io
import numpy as np

# Define the video path
video_path = "/ssd_scratch/cvit/vanshg/gridcorpus/video/s1/srbizp.mpg"

# OpenCV Benchmark
start_time = time.time()
cap = cv2.VideoCapture(video_path)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
frames = np.array(frames)
print(f"frames.shape = {frames.shape}")
opencv_duration = time.time() - start_time
print(f"OpenCV read duration: {opencv_duration:.2f} seconds, total frames: {len(frames)}")

# Torchvision Benchmark
start_time = time.time()
video, audio, info = io.read_video(video_path)
print(f"vide.shape = {video.shape}")
torchvision_duration = time.time() - start_time
print(f"Torchvision read duration: {torchvision_duration:.2f} seconds, total frames: {video.shape[0]}")

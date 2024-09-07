import os
import glob
import cv2
from tqdm import tqdm

data_dir = "/ssd_scratch/cvit/vanshg/datasets/accented_speakers"
speaker = "diane_jennings"
speaker_dir = os.path.join(data_dir, speaker)
src_vid_dir = os.path.join(speaker_dir, f"processed_videos")

# Initialize total duration
total_duration = 0

# Get list of mp4 files
video_files = glob.glob(f"{src_vid_dir}/*.mp4")
# video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]

# Iterate through all files with a progress bar
for file_path in tqdm(video_files, desc="Processing videos", unit="video"):
    cap = cv2.VideoCapture(file_path)
    
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = total_frames/fps
    total_duration += duration

# Convert total duration to hours, minutes, seconds
hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)
seconds = int(total_duration % 60)

# Print the total duration
print(f"\nTotal duration of all videos: {hours} hours, {minutes} minutes, {seconds} seconds")
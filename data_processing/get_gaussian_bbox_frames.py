import math
import os
import cv2
import numpy as np
import glob
from PIL import Image

from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube',
    help="Directory of original dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='mia_sandra',
    help='Name of speaker'
)
parser.add_argument(
    '--num-jobs',
    help='Number of processes (jobs) across which to run in parallel',
    default=4,
    type=int
)
parser.add_argument(
    '--job-index',
    type=int,
    default=3,
    help='Index to identify separate jobs (useful for parallel processing)'
)
args = parser.parse_args()

speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(speaker_dir, "videos")

dst_crops_dir = os.path.join(speaker_dir, "gaussian_bbox_frames")

def process_video_file(video_path, args, video_id):
    print(f"Processing {video_id = } | {video_path = }")
    video_name = os.path.basename(video_path).split('.')[0]
    dst_vid_crops_dir = os.path.join(dst_crops_dir, video_name)
    os.makedirs(dst_vid_crops_dir, exist_ok=True)

    # For 720p
    x, y, w, h = 5, 582, 1266, 135

    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"{total_frames = }")

    # Process each frame in the video
    frame_count = 0
    with tqdm(total=total_frames, desc=f"Processing Video {video_id} Frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the frame to the bounding box
            cropped_frame = frame[y:y+h, x:x+w]

            # Convert the cropped frame to grayscale
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            # Create a binary mask where white areas are white (255) and others are black (0)
            _, mask = cv2.threshold(gray_frame, 240, 255, cv2.THRESH_BINARY)

            # Apply the mask to the cropped frame
            # white_parts = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
            white_parts = np.copy(gray_frame)

            white_parts[gray_frame >= 240] = 255
            white_parts[gray_frame < 240] = 0
            # white_parts = cv2.GaussianBlur(white_parts, (3, 3), 0)

            # Save every frame
            frame_filename = os.path.join(dst_vid_crops_dir, f"frame_{frame_count:05d}.png")
            # print(f"{frame_count = } | {white_parts.shape = } | {frame_filename = }")
            cv2.imwrite(frame_filename, white_parts)

            frame_count += 1
            pbar.update(1)

        # Release the video capture object
        cap.release()

def main(args):
    video_ids_file = os.path.join(speaker_dir, "new_video_ids2.txt")
    video_ids = open(video_ids_file, 'r').read().split()
    print(f"{video_ids = }")
    video_files = [os.path.join(src_vid_dir, f"{video_id}.mkv") for video_id in video_ids]

    # video_files = glob.glob(os.path.join(src_vid_dir, "*.mkv"))
    # video_files = [os.path.join(src_vid_dir, "qAt94Wmcavw.mkv")]
    video_files = sorted(video_files)
    print(f"Total number of video_files: {len(video_files)}")

    unit = math.ceil(len(video_files) * 1.0 / args.num_jobs)
    video_files = video_files[args.job_index * unit : (args.job_index + 1) * unit]
    print(f"Number of files for job {args.job_index}/{args.num_jobs} index: {len(video_files)}")

    for i, video_path in enumerate(tqdm(video_files, desc=f"Processing Video")):
        process_video_file(video_path, args, video_id=i)
    
    print(f"WROTE THE GAUSSIAN BBOX FRAMES OF ALL VIDEOS!!!")

if __name__ == '__main__':
    main(args)
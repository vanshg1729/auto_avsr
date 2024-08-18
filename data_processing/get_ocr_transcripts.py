import os
import sys
import glob
import logging

from tqdm import tqdm
import numpy as np
import cv2
import argparse

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from paddleocr import PaddleOCR

# Set the logging level to WARNING to suppress DEBUG and INFO messages
# logging.getLogger().setLevel(logging.WARNING)
# Configure logging for PaddleOCR
# logging.getLogger('paddleocr').setLevel(logging.WARNING)
# logging.getLogger('paddle').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='./datasets/deaf-youtube',
    help="Directory of original dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='mia_sandra',
    help='Name of speaker'
)
parser.add_argument(
    '--num-workers',
    help='Number of processes (jobs) across which to run in parallel',
    default=1,
    type=int
)
args = parser.parse_args()

src_speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(src_speaker_dir, "videos")
src_frames_dir = os.path.join(src_speaker_dir, "extracted_frames")
dst_ocr_dir = os.path.join(src_speaker_dir, "ocr_transcripts")

def process_frame(ocr, frame, frame_idx, video_path, num_frames, output_dir):
    padding = 5
    frame_name = f"frame_{str(frame_idx).zfill(padding)}.txt"
    video_name = os.path.basename(video_path).split('.')[0]
    # print(f"Processing frame {frame_idx} of video {video_name}")

    dst_vid_ocr_dir = os.path.join(output_dir, video_name)
    os.makedirs(dst_vid_ocr_dir, exist_ok=True)

    output_file = os.path.join(dst_vid_ocr_dir, frame_name)

    results = ocr.ocr(frame, cls=True)
    confidence = 1.0
    if results and results[0]:
        transcribed_text = ' '.join([result[1][0] for result in results[0]])
        confidence = min([result[1][1] for result in results[0]])
    else:
        transcribed_text = None

    with open(output_file, 'w') as f:
        f.write(f"{transcribed_text} {confidence}")

def worker(input_queue, progress_bar):
    ocr = PaddleOCR(lang='en', use_angle_cls=True)
    while True:
        data = input_queue.get()
        if data is None: # Stop Signal
            break
        
        frame, frame_idx, video_path, num_frames = data
        process_frame(ocr, frame, frame_idx, video_path, num_frames, dst_ocr_dir)
        progress_bar.update(1) # Update the progress bar after processing each frame

def main(args):
    num_workers = args.num_workers

    video_path = os.path.join(src_vid_dir, "3aAi2dqQ1iI.mp4")
    print(f"{video_path = }")
    video_name = os.path.basename(video_path)
    print(f"{os.path.exists(video_path) = }")
    # cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames to calculate padding
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a progress bar
    progress_bar = tqdm(total=total_frames, 
                        desc=f"Processing frames of {video_name}", unit="frame")

    # Create an input queue for communication between processes
    input_queue = mp.Queue()

    # Start worker processes
    workers = []
    for _ in range(args.num_workers):
        p = mp.Process(target=worker, args=(input_queue, progress_bar))
        p.start()
        workers.append(p)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        input_queue.put((frame, frame_idx, video_path, total_frames))
        frame_idx += 1
    
    # Send stop signals to workers
    for _ in range(num_workers):
        input_queue.put(None)

    # Clean up
    for p in workers:
        p.join()
    
    cap.release()

if __name__ == '__main__':
    main(args)
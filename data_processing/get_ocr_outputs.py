import os
import sys
import glob
import time

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
# from concurrent.futures import ThreadPoolExecutor, as_completed
print(f"Before importing Paddle: {time.time()}")
from paddleocr import PaddleOCR
print(f"After importing Paddle: {time.time()}")

from tqdm import tqdm
import cv2
import numpy as np

import argparse
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
    default=4,
    type=int
)
args = parser.parse_args()

src_speaker_dir = os.path.join(args.data_dir, args.speaker)

src_vid_dir = os.path.join(src_speaker_dir, "videos")
src_crops_dir = os.path.join(src_speaker_dir, "gaussian_bbox_frames")
dst_ocr_dir = os.path.join(src_speaker_dir, "ocr_outputs")

ocr = PaddleOCR(lang='en', use_angle_cls=True)
# ocr = PaddleOCR(lang='en')

def process_frame(ocr, frame_path, frame_idx, video_path, output_dir):
    padding = 5
    frame_name = f"frame_{str(frame_idx).zfill(padding)}"
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"Processing frame {frame_idx} of video {video_name}")

    dst_vid_ocr_dir = os.path.join(output_dir, video_name)
    os.makedirs(dst_vid_ocr_dir, exist_ok=True)

    output_file = os.path.join(dst_vid_ocr_dir, f"{frame_name}.txt")

    results = ocr.ocr(frame_path, cls=True)
    confidence = 1.0
    if results and results[0]:
        transcribed_text = ' '.join([result[1][0] for result in results[0]])
        confidence = min([result[1][1] for result in results[0]])
    else:
        transcribed_text = None

    with open(output_file, 'w') as f:
        f.write(f"{transcribed_text} {confidence}")
    
    return transcribed_text

def worker(input_queue, worker_id, progress_bar, progress_lock):
    print(f"Inside worker with ID: {worker_id}")
    # ocr = PaddleOCR(lang='en', use_angle_cls=True)
    while True:
        try:
            data = input_queue.get()
            if data is None: # Stop Signal
                break
            
            frame, frame_idx, video_path, num_frames = data
            process_frame(ocr, frame, frame_idx, video_path, dst_ocr_dir)

            # with progress_lock:
            #     progress_bar.update(1) # Update the progress bar after processing each frame
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")

def main():
    num_workers = args.num_workers
    print(f"Num workers: {num_workers}")
    # video_files = glob.glob(os.path.join(src_vid_dir, f"*.mkv"))
    video_files = [os.path.join(src_vid_dir, f"qAt94Wmcavw.mkv")]

    for video_path in video_files:
        video_name = os.path.basename(video_path).split('.')[0]
        src_vid_crops_dir = os.path.join(src_crops_dir, f"{video_name}")
        video_path = os.path.join(src_vid_dir, f"{video_name}.mkv")
        print(f"{video_path = }")
        print(f"{os.path.exists(video_path) = }")
        assert os.path.exists(video_path), f"{video_path} does not exists"

        # Get the total number of frames to calculate padding
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_filepaths = glob.glob(os.path.join(src_vid_crops_dir, "*.png"))
        frame_filepaths = sorted(frame_filepaths)
        frame_filepaths = frame_filepaths[:3500]
        print(f"Number of Frame Filepaths: {len(frame_filepaths)}")

        # Create an input queue for communication between processes
        input_queue = mp.Queue()

        # Start worker processes
        workers = []
        try:
            for i in range(args.num_workers):
                p = mp.Process(target=worker, args=(input_queue, i, None, None))
                p.start()
                workers.append(p)

            for frame_idx, frame_path in enumerate(frame_filepaths):
                if frame_idx > 3500:
                    break
                print(frame_path)
                frame_filename = os.path.basename(frame_path).split('.')[0]
                frame_idx = int(frame_filename.split('_')[1])
                input_queue.put((frame_path, frame_idx, video_path, total_frames))

            # Send stop signals to workers
            for _ in range(num_workers):
                input_queue.put(None)
        except KeyboardInterrupt:
            print(f"Interrupted! Cleaning up processes...")
            for p in workers:
                p.terminate()
                p.join() # Ensure all processes are cleaned up
            sys.exit(1)
        finally:
            for p in workers:
                p.join()

        print(f"FINISHED GETTING OCR OUTPUTS OF VIDEO: {video_path}")
    
    print(f"FINISHED PROCESSING ALL FRAMES OF SPEAKER: {args.speaker}")

if __name__ == '__main__':
    main()
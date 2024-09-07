import argparse
import glob
import json
import string
import math
import os
import pickle
import shutil
import warnings
import sys
from tqdm import tqdm

import cv2
import numpy as np
import torch

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback, subprocess

from video_utils import clip_video_ffmpeg

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument Parsing
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube',
    help="Directory of original dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    choices=['retinaface', 'yolov5', 'face_alignment'],
    help="Type of face detector. (Default: retinaface)",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='deafdaydreamer',
    help='Name of speaker'
)
parser.add_argument(
    '--ngpu',
    help='Number of GPUs across which to run in parallel',
    default=3,
    type=int
)
parser.add_argument(
    '--job-index',
    help='Job id for splitting the processing',
    default=0,
    type=int
)
parser.add_argument(
    '--batch-size',
    help='Single GPU Face Detection batch size',
    default=32,
    type=int
)

args = parser.parse_args()
print(f"Detecting faces using : {args.detector}")

src_speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(src_speaker_dir, "videos")
dst_vid_dir = os.path.join(src_speaker_dir, "face_tracks")
print(f"Src video dir = {src_vid_dir}")
print(f"DST vid dir = {dst_vid_dir}")

gpu_id = args.job_index
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor
model_name = "resnet50"
face_detector = RetinaFacePredictor(device=f"cuda:{gpu_id}", threshold=0.8, 
                                     model=RetinaFacePredictor.get_model(model_name))

def get_batch_prediction_retinaface(frames, face_detector):
    preds = []
    for frame in frames:
         pred = face_detector(frame)
         preds.append(pred)
    
    return preds

def save_track(video_path, track, output_path, fps):
    start_frame = track['start_frame']
    end_frame = track['end_frame']
    num_frames = end_frame - start_frame + 1

    start_time = start_frame/fps
    end_time = end_frame/fps
    timestamp = (start_time, end_time)

    # Don't save the video if it is less than 1 second
    if num_frames < fps:
        print(f"\nvideo track is less than 1 second: {num_frames = } | {start_frame = } | {end_frame = }")
        return {}

    print(f"\nStart Frame: {start_frame} | End Frame: {end_frame}")
    clip_video_ffmpeg(video_path, timestamp, output_path, verbose=True)
    track_metadata = {'input_path': video_path, 'output_path': output_path,
                      'start_time': start_time, 'end_time': end_time, 'fps': fps,
                      "start_frame": start_frame, "end_frame": end_frame}
    print(f"Saved the face track with {num_frames = } to {output_path}")

    return track_metadata

def video_frame_batch_generator(video_path, batch_size):
    """
    Generator function that reads frames from a video and yields them in batches.

    Args:
        video_path (str): Path to the video file.
        batch_size (int): Number of frames per batch.

    Yields:
        tuple: A tuple containing a list of frames and a list of frame indices.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    batch_frames = []
    batch_indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # coverting to RGB
        batch_frames.append(frame)
        batch_indices.append(frame_idx)
        frame_idx += 1

        if len(batch_frames) == batch_size:
            yield batch_frames, batch_indices
            batch_frames = []
            batch_indices = []

    # Yield the last batch if it's not empty
    if batch_frames:
        yield batch_frames, batch_indices

    cap.release()

def process_video_file(video_path, args, gpu_id=0, video_id=0):
    print(f"Processing video: {video_path} with gpu {gpu_id}")
    
    # Getting the output clips directory for this video
    video_fname = os.path.basename(video_path).split('.')[0]
    clips_dir = os.path.join(dst_vid_dir, f"{video_fname}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get the total number of frames and fps
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Number of Frames: {total_frames} | FPS: {fps}")
    batch_size = args.batch_size

    # FACE TRACKING
    track_id = 0
    tracks = []
    tracks_metadata = []
    metadata_filepath = os.path.join(clips_dir, f"tracks.json")
    video_loader = video_frame_batch_generator(video_path, batch_size)

    # Batch Processing of Frames
    for frames, frame_ids in tqdm(video_loader, total=total_frames//batch_size, 
                                  desc=f"Processing video {video_id} Frame Batches"):
        preds = get_batch_prediction_retinaface(frames, face_detector)

        # process each frame individually to get face tracks
        for i in range(len(frames)):
            # Continue if not face is detected in the frame
            if (preds[i] is None) or (len(preds[i]) == 0):
                continue
            
            frame, frame_idx = frames[i], frame_ids[i]

            # Detected face along with bounding box
            # detected_face = detected_faces[0]
            # (x1, y1, x2, y2) = detected_face[:4]
            # w, h = (x2 - x1), (y2 - y1)
            # bbox = (x1, y1, w, h)

            # Check the already existing tracks
            create_new_track = True
            if len(tracks):
                last_track = tracks[-1]
                last_track_frame = last_track['end_frame']

                # Continue the previous track
                if frame_idx == last_track_frame + 1:
                    last_track['end_frame'] = frame_idx
                    # last_track['frames'].append(frame)
                    # last_track['bboxes'].append(bbox)
                    create_new_track = False
            
            # Start a new track
            if create_new_track:
                # Save the previous track if there is one
                if len(tracks):
                    prev_track = tracks[-1]
                    out_vid_path = os.path.join(clips_dir, f"track-{track_id}.mkv")
                    track_metadata = save_track(video_path, prev_track, out_vid_path, fps)
                    if len(track_metadata):
                        tracks_metadata.append(track_metadata)
                        track_id += 1

                    tracks = [] # empty the previous tracks array

                new_track = {
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    # "frames": [frame],
                    # "bboxes": [bbox]
                }
                tracks.append(new_track)
                print(f"\nStarted a new track at frame {frame_idx}")
    
    # Save the last track if there is one left
    if len(tracks):
        prev_track = tracks[-1]
        out_vid_path = os.path.join(clips_dir, f"track-{track_id}.mkv")
        track_metadata = save_track(video_path, prev_track, out_vid_path, fps)
        if len(track_metadata):
            tracks_metadata.append(track_metadata)
            track_id += 1

        tracks = [] # empty the previous tracks array
    
    # Save the tracks.json metadata file after all tracks have been detected
    with open(metadata_filepath, 'w') as json_file:
        json.dump(tracks_metadata, json_file, indent=4)
        print(f"Saved the tracks metadata to {metadata_filepath}")

def main(args):
    # video_ids_file = os.path.join(src_speaker_dir, "new_video_ids2.txt")
    # video_ids = open(video_ids_file, 'r').read().split()
    # print(f"{video_ids = }")
    # video_files = [os.path.join(src_vid_dir, f"{video_id}.mkv") for video_id in video_ids]

    video_files = glob.glob(os.path.join(src_vid_dir, "*.mkv"))
    video_files = sorted(video_files)
    print(f"Total number of Video Files: {len(video_files)}")
    print(f"{video_files[0] = }")

    for video_idx, video_path in enumerate(tqdm(video_files, desc="Processing Video")):
        process_video_file(video_path, args, gpu_id=gpu_id, video_id=video_idx)

    print(f"WROTE FACE_TRACKS OF ALL VIDEOS OF SPEAKER {args.speaker} !!!")

if __name__ == '__main__':
    main(args)
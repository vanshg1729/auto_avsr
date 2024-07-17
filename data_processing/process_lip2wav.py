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
    default='./datasets/Lip2Wav',
    help="Directory of original dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Type of face detector. (Default: retinaface)",
)
parser.add_argument(
    "--root-dir",
    type=str,
    default='./datasets/Lip2Wav',
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='hs',
    help='Name of speaker'
)
parser.add_argument(
    '--ngpu',
    help='Number of GPUs across which to run in parallel',
    default=1,
    type=int
)

def save_track(video_path, track, output_path, fps):
    start_frame = track['start_frame']
    end_frame = track['end_frame']
    num_frames = end_frame - start_frame + 1

    start_time = int(start_frame/fps)
    end_time = int(end_frame/fps) + 1
    timestamp = (start_time, end_time)

    # Don't save the video if it is less than 1 second
    if num_frames < fps:
        print(f"video track is less than 1 second: {num_frames = } | {start_frame = } | {end_frame = }")
        return {}

    clip_video_ffmpeg(video_path, timestamp, output_path)
    track_metadata = {'input_path': video_path, 'output_path': output_path,
                      'start_time': start_time, 'end_time': end_time, 'fps': fps,
                      "start_frame": start_frame, "end_frame": end_frame}
    print(f"Saved the face track with {num_frames = } to {output_path}")

    return track_metadata

args = parser.parse_args()

src_speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(src_speaker_dir, "videos")
dst_vid_dir = os.path.join(src_speaker_dir, "face_tracks")
print(f"Src video dir = {src_vid_dir}")
print(f"DST vid dir = {dst_vid_dir}")

video_files = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
video_files = sorted(video_files)
print(f"Total number of Video Files: {len(video_files)}")
print(f"{video_files[0] = }")

# RetinaFace detector
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor
model_name = "resnet50"
face_detectors = [RetinaFacePredictor(device=f"cuda:{i}", threshold=0.8,
                                    model=RetinaFacePredictor.get_model(model_name))
                                    for i in range(args.ngpu)]

def process_video_file(video_path, args, gpu_id=0):
    print(f"Processing video: {video_path}")
    face_detector = face_detectors[gpu_id]
    
    # Getting the output clips directory for this video
    video_fname = os.path.basename(video_path).split('.')[0]
    clips_dir = os.path.join(dst_vid_dir, f"{video_fname}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get the total number of frames and fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Number of Frames: {frame_count} | FPS: {fps}")

    track_id = 0
    # FACE TRACKING
    tracks = []
    tracks_metadata = []
    metadata_filepath = os.path.join(clips_dir, f"tracks.json")

    for frame_idx in tqdm(range(frame_count), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            print(f"No more frames to process in {video_path}")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = face_detector(frame)

        # continue if no face is detected in the frame
        if len(detected_faces) == 0:
            continue

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
                out_vid_path = os.path.join(clips_dir, f"track-{track_id}.mp4")
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
        out_vid_path = os.path.join(clips_dir, f"track-{track_id}.mp4")
        track_metadata = save_track(video_path, prev_track, out_vid_path, fps)
        if len(track_metadata):
            tracks_metadata.append(track_metadata)

        track_id += 1
        tracks = [] # empty the previous tracks array
    
    with open(metadata_filepath, 'w') as json_file:
        json.dump(tracks_metadata, json_file, indent=4)
        print(f"Saved the tracks metadata to {metadata_filepath}")

def mp_handler(job):
    video_path, args, gpu_id = job
    try:
        process_video_file(video_path, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)

def main(args):
    video_files = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
    video_files = sorted(video_files)
    print(f"Total number of Video Files: {len(video_files)}")
    print(f"{video_files[0] = }")

    jobs = [(video_path, args, i % args.ngpu) for i, video_path in enumerate(video_files)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    # for idx in tqdm(range(len(video_files)), desc="Processing Videos"):
    #     video_path = video_files[idx]
    #     process_video_file(video_path, args)

if __name__ == '__main__':
    main(args)
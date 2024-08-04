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
import shutil

import cv2
import numpy as np
import torch

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback, subprocess

from video_utils import clip_video_ffmpeg, align_track_to_segments, save_track_clips

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument Parsing
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/',
    help="Directory of original dataset",
)
parser.add_argument(
    "--root-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube',
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='benny-large',
    help='Name of speaker'
)

args = parser.parse_args()

src_speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(src_speaker_dir, "videos")
src_tracks_dir = os.path.join(src_speaker_dir, "face_tracks")
src_segments_dir = os.path.join(src_speaker_dir, "captions")

dst_clips_dir = os.path.join(src_speaker_dir, "sentence_clips")

print(f"Src Video Dir: {src_vid_dir}, {os.path.exists(src_vid_dir)}")
print(f"Src Tracks Dir: {src_tracks_dir}")
print(f"Src Segments Dir: {src_segments_dir}")
print(f"Dst Clips Dir: {dst_clips_dir}")

video_files = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
video_files = sorted(video_files)
print(f"Total number of Video Files: {len(video_files)}")
print(f"{video_files[0] = }")

for video_idx, video_file in enumerate(tqdm(video_files, desc="Making Video Clips")):
    video_fname = os.path.basename(video_file).split('.')[0]
    print(f"Processing video {video_idx} with path: {video_file}")

    # Get all the face tracks for this video
    video_tracks_path = os.path.join(src_tracks_dir, f"{video_fname}/tracks.json")
    tracks_list = json.load(open(video_tracks_path))

    # Get the Aligned Sentence Segments for this video
    video_segments_path = os.path.join(src_segments_dir, f"{video_fname}.json")
    aligned_segments = json.load(open(video_segments_path))

    tracks_metadata = []
    total_clips = 0
    # Aligning the Face track to Sentence Segments and saving them
    for track_id, track in enumerate(tracks_list):
        # Get the clips for each of those tracks
        track_clips = align_track_to_segments(track, aligned_segments, word_level=False)

        # Save clips for each track
        track_metadata = save_track_clips(
            track,
            track_id,
            track_clips,
            src_vid_dir,
            dst_clips_dir,
            roundoff=True
        )

        tracks_metadata.append(track_metadata)
        total_clips += len(track_metadata['clips'])
    
    print(f"Total Number of clips : {total_clips}")

    # Saving the tracks metadata for this video
    video_clips_dir = os.path.join(dst_clips_dir, f"{video_fname}")
    track_metadata_path = os.path.join(video_clips_dir, "clips.json")
    with open(track_metadata_path, 'w') as json_file:
        json.dump(tracks_metadata, json_file)
    
    # Copying the face tracks and aligned segments file
    shutil.copy(video_tracks_path, video_clips_dir)
    shutil.copy(video_segments_path, video_clips_dir)
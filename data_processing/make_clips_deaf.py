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
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube',
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
    default='jazzy',
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

min_clip_duration = 0.0
src_speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(src_speaker_dir, "videos")
src_tracks_dir = os.path.join(src_speaker_dir, "face_tracks")
src_segments_dir = os.path.join(src_speaker_dir, "captions")

dst_clips_dir = os.path.join(src_speaker_dir, "sentence_clips")
# dst_clips_dir = src_speaker_dir

print(f"Src Video Dir: {src_vid_dir}, {os.path.exists(src_vid_dir)}")
print(f"Src Tracks Dir: {src_tracks_dir}")
print(f"Src Segments Dir: {src_segments_dir}")
print(f"Dst Clips Dir: {dst_clips_dir}")

def process_video_file(video_path, args, job_id=0, video_id=0):
    video_fname = os.path.basename(video_path).split('.')[0]
    print(f"Processing video {video_id} with path: {video_path}")

    # Get all the face tracks for this video
    video_tracks_path = os.path.join(src_tracks_dir, f"{video_fname}/tracks.json")
    tracks_list = json.load(open(video_tracks_path))

    # Get the Aligned Sentence Segments for this video
    video_segments_path = os.path.join(src_segments_dir, f"{video_fname}.json")
    aligned_segments = json.load(open(video_segments_path))

    tracks_metadata = []
    total_clips = 0
    clip_id = 0
    # Aligning the Face track to Sentence Segments and saving them
    for track_id, track in enumerate(tracks_list):
        # Get the clips for each of those tracks
        track_clips = []
        aligned_track_clips = align_track_to_segments(track, aligned_segments, min_clip_len=0.5, word_level=False)

        # Go through each of the segment clips aligned to the face track
        for clip in aligned_track_clips:
            clip_st = clip['start']
            clip_end = clip['end']
            clip_duration = clip_end - clip_st
            # continue if clip is too small
            if clip_duration < min_clip_duration:
                continue
            # store the clip
            clip['clip_id'] = clip_id
            track_clips.append(clip)
            clip_id += 1

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
    if not os.path.exists(video_clips_dir):
        print(f"There are no clips for video {video_fname}")
    os.makedirs(video_clips_dir, exist_ok=True)
    track_metadata_path = os.path.join(video_clips_dir, "clips.json")
    with open(track_metadata_path, 'w') as json_file:
        json.dump(tracks_metadata, json_file)
    
    # Copying the face tracks and aligned segments file
    shutil.copy(video_tracks_path, video_clips_dir)
    shutil.copy(video_segments_path, video_clips_dir)

def main(args):
    # Read the list of videos from videos.txt instead
    # video_ids_file = os.path.join(src_speaker_dir, "copy_all_video_ids.txt")
    # video_ids = open(video_ids_file, 'r').read().split()
    # print(f"{video_ids = }")
    # video_files = [os.path.join(src_vid_dir, f"{video_id}.mp4") for video_id in video_ids]

    video_files = glob.glob(os.path.join(src_vid_dir, "*.mkv"))
    video_files = sorted(video_files)
    # video_files = [os.path.join(src_vid_dir, "_0MutuU6eks.mp4")]
    print(f"Total number of Video Files: {len(video_files)}")
    print(f"{video_files[0] = }")

    unit = math.ceil(len(video_files) * 1.0 / args.num_jobs)
    video_files = video_files[args.job_index * unit : (args.job_index + 1) * unit]
    print(f"Number of files for job {args.job_index}/{args.num_jobs} index: {len(video_files)}")

    for i, video_path in enumerate(tqdm(video_files, desc=f"Processing Video")):
        process_video_file(video_path, args, job_id=args.job_index, video_id=i)
    
    print(f"WROTE SENTENCE_CLIPS ALL VIDEOS in job {args.job_index}/{args.num_jobs} OF SPEAKER {args.speaker}!!!")

if __name__ == '__main__':
    main(args)
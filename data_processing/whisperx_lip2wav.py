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

import whisperx

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument Parsing
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/accented_speakers',
    help="Directory of original dataset",
)
parser.add_argument(
    "--model",
    type=str,
    default="large-v3",
    choices=['medium', 'large-v2', 'large-v3'],
    help="Type of WhisperX model",
)
parser.add_argument(
    "--model-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/checkpoints',
    help='Directory to store WhisperX model checkpoint'
)
parser.add_argument(
    '--speaker',
    type=str,
    default='crazy_russian',
    help='Name of speaker'
)
parser.add_argument(
    '--ngpu',
    help='Number of GPUs across which to run in parallel',
    default=1,
    type=int
)
parser.add_argument(
    '--batch-size',
    help='Single GPU WhisperX trascribe batch size',
    default=2,
    type=int
)

args = parser.parse_args()
src_speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(src_speaker_dir, "videos")
dst_text_dir = os.path.join(src_speaker_dir, "whisperx_transcripts")
print(f"Src video dir = {src_vid_dir}")
print(f"DST transcription dir = {dst_text_dir}")

# Loading the WhisperX and Wav2Vec2.0 Alignment models
compute_type = "float16"
whisperx_model_dir = os.path.join(args.model_dir, "whisperx")
os.makedirs(whisperx_model_dir, exist_ok=True)
model = whisperx.load_model(args.model, device=f'cuda', compute_type=compute_type, 
                            download_root=whisperx_model_dir, language='en', threads=4)

model_a, metadata = whisperx.load_align_model(language_code='en', 
                                          device=f'cuda', model_dir=args.model_dir)
print(f"Loaded the WhisperX model and Wav2vec2 alignment model")

def transcribe_video_file(video_path, args, gpu_id=0, video_id=0):
    batch_size = args.batch_size 

    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=batch_size)
    print(f"\nGot the Result for video {video_id} with {video_path = }")
    
    aligned_result = whisperx.align(result['segments'], model_a, metadata, 
                                    audio, device, return_char_alignments=False)
    print(f"Got the Aligned Result for {video_id} with {video_path = }")
    
    # Writing the transcriptions
    vid_fname = os.path.basename(video_path).split('.')[0]
    vid_text_dir = os.path.join(dst_text_dir, vid_fname)
    os.makedirs(vid_text_dir, exist_ok=True)
    segments_file = os.path.join(vid_text_dir, "segments.json")
    aligned_segments_file = os.path.join(vid_text_dir, "aligned_segments.json")
    word_segments_file = os.path.join(vid_text_dir, "word_segments.json")

    with open(segments_file, 'w') as json_file:
        json.dump(result['segments'], json_file)
        print(f"\nWrote segments for video {video_id} {video_path = } to {segments_file}")

    with open(aligned_segments_file, 'w') as json_file:
        json.dump(aligned_result['segments'], json_file)
        print(f"Wrote aligned segments for video {video_id} {video_path = } to {aligned_segments_file}")

    with open(word_segments_file, 'w') as json_file:
        json.dump(aligned_result['word_segments'], json_file)
        print(f"Wrote word segments for video {video_id} {video_path = } to {word_segments_file}")

def mp_handler(job):
    video_path, args, gpu_id, video_id = job
    try:
        transcribe_video_file(video_path, args, gpu_id, video_id)
    except KeyboardInterrupt:
        exit(0)
    
def main(args):
    # video_ids_file = os.path.join(src_speaker_dir, "all_video_ids.txt")
    # video_ids = open(video_ids_file, 'r').read().split()
    # print(f"{video_ids = }")
    # video_files = [os.path.join(src_vid_dir, f"{video_id}.mp4") for video_id in video_ids]

    video_files = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
    # video_files = [os.path.join(src_vid_dir, "EiEIfBatnH8_crop.mp4")]
    video_files = sorted(video_files)
    print(f"Total number of Video Files: {len(video_files)}")
    print(f"{video_files[0] = }")

    for video_id, video_file in enumerate(tqdm(video_files, desc="Transcribing Videos")):
        transcribe_video_file(video_file, args, video_id=video_id)

    print(f"WROTE TRANSCRIPTS FOR ALL VIDEOS OF SPEAKER {args.speaker} !!!")

if __name__ == '__main__':
    main(args)
import argparse
import glob
import math
import os
import pickle
import shutil
import warnings
import string

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

data_dir = "/ssd_scratch/cvit/vanshg/datasets/deaf-youtube"
speaker_name = "benny"

src_speaker_dir = os.path.join(data_dir, f"{speaker_name}")
src_video_dir = os.path.join(src_speaker_dir, "processed_videos")
src_text_dir = os.path.join(src_speaker_dir, f"transcriptions")

print(f"SRC Speaker DIR: {src_speaker_dir}, {os.path.exists(src_speaker_dir)}")
print(f"SRC Video DIR: {src_video_dir}, {os.path.exists(src_video_dir)}")
print(f"SRC Text DIR: {src_text_dir}, {os.path.exists(src_text_dir)}")

def process_text(text):
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.upper()
    return text

def get_gt_text(file_path):
    src_txt_filename = file_path
    with open(src_txt_filename, "r") as file:
        text = file.read()
    
    return text

video_files = glob.glob(os.path.join(src_video_dir, "*.mp4"))
video_files = sorted(video_files)
print(f"{len(video_files) = }")
print(f"{video_files[0] = }")

# Label file for speaker
dst_label_file = os.path.join(src_speaker_dir, "labels.txt")
f = open(dst_label_file, "w")
print(f"DST Label File: {dst_label_file}")

for video_idx, video_file in enumerate(tqdm(video_files, desc="Creating Labels for videos")):
    video_fname = os.path.basename(video_file).split('.')[0]

    dst_txt_filename = os.path.join(src_text_dir, f"{video_fname}.txt")
    gt_text = get_gt_text(dst_txt_filename)

    basename = f"processed_videos/{video_fname}.mp4"
    f.write(
        f"{basename} {gt_text}\n"
    )
    print(f"{basename} {gt_text}")

f.close()
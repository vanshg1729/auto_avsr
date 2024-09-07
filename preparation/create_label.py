import argparse
import glob
import math
import os
import pickle
import shutil
import warnings
import string
import cv2

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

data_dir = "/ssd_scratch/cvit/vanshg/datasets/accented_speakers"
speaker_name = "diane_jennings"

src_speaker_dir = os.path.join(data_dir, f"{speaker_name}")
src_video_dir = os.path.join(src_speaker_dir, f"processed_videos")
src_text_dir = os.path.join(src_speaker_dir, f"transcriptions")

print(f"SRC Speaker DIR: {src_speaker_dir}, {os.path.exists(src_speaker_dir)}")
print(f"SRC Video DIR: {src_video_dir}, {os.path.exists(src_video_dir)}")
print(f"SRC Text DIR: {src_text_dir}, {os.path.exists(src_text_dir)}")

assert os.path.exists(src_speaker_dir), f"{src_speaker_dir = } doesn't exists"
assert os.path.exists(src_video_dir), f"{src_video_dir = } doesn't exists"
assert os.path.exists(src_text_dir), f"{src_text_dir = } doesn't exists"

def process_text(text):
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.upper()
    return text

def get_gt_text(file_path):
    src_txt_filename = file_path
    with open(src_txt_filename, "r", encoding='utf-8') as file:
        text = file.read()
    
    return text

# Creating from video ids
# video_ids_file = os.path.join(src_speaker_dir, "train_new.txt")
# video_ids = open(video_ids_file, 'r').read().split()
# video_files = []
# print(f"{video_ids = }")
# for video_id in video_ids:
#     video_files += glob.glob(os.path.join(src_video_dir, f"{video_id}*.mp4"))
# print(len(video_files))
# video_files = sorted(video_files)

video_files = glob.glob(os.path.join(src_video_dir, "*.mp4"))
# video_files = glob.glob(os.path.join(src_speaker_dir, "sentence_clips/*/*.mp4"))
# video_files = glob.glob(os.path.join(src_video_dir, "EiEIfBatnH8_crop*.mp4"))
# video_files = sorted(video_files)

print(f"{len(video_files) = }")
print(f"{video_files[0] = }")

# Label file for speaker
dst_label_file = os.path.join(src_speaker_dir, "all_labels.txt")
f = open(dst_label_file, "w")
print(f"DST Label File: {dst_label_file}")

for video_idx, video_file in enumerate(tqdm(video_files, desc="Creating Labels for videos")):
    cap = cv2.VideoCapture(video_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    video_fname = os.path.basename(video_file).split('.')[0]

    src_txt_filename = os.path.join(src_text_dir, f"{video_fname}.txt")
    gt_text = get_gt_text(src_txt_filename)
    gt_len = len(gt_text.split())

    if total_frames > 500 or total_frames <= 0:
        print(f"File {video_idx = } {video_file} with has {total_frames = }")
        # continue

    if gt_len == 0:
        print(f"File {video_idx = } {video_file} {gt_text = } HAS 0 GT LEN")
        continue

    basename = os.path.relpath(video_file, start=data_dir)
    f.write(
        f"{basename} {gt_text}\n"
    )
    # print(f"{basename} {gt_text}")

f.close()
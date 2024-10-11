# This file is used to create a reduced label set using the existing labels based on the highest WER sentences
import json
import random
import os
import math
import numpy as np
import cv2
import pandas as pd
import copy

def seconds_to_hhmmss(seconds):
    """
    Convert seconds to hh:mm:ss format.
    
    Args:
        seconds (float): Time in seconds.
    
    Returns:
        str: Time in hh:mm:ss format.
    """
    seconds = math.ceil(seconds * 1e3)/1e3 # Rounding to the nearest decimal of 1e-3
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    # print(f"{seconds = } | {hours = } | {minutes = } | {secs = }")
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

seed = 40

data_dir = "/ssd_scratch/cvit/vanshg/datasets/deaf-youtube"
speaker_name = "deafdaydreamer"
speaker_dir = os.path.join(data_dir, f"{speaker_name}")

train_label_filepath = os.path.join(speaker_dir, "train_labels.txt")
label_lines = open(train_label_filepath, 'r').readlines()

# Randomly shuffling the train labels 
rng = random.Random(seed)
rng.shuffle(label_lines)

# List of reduced labels
reduced_labels = []
max_seconds = 2400 # 50 mins
total_seconds = 0

for i, line in enumerate(label_lines):
    rel_video_path, gt_text = line.split()[0], ' '.join(line.split()[1:])
    video_path = os.path.join(data_dir, rel_video_path)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_in_seconds = total_frames / fps
    if total_seconds <= max_seconds:
        total_seconds += duration_in_seconds
        reduced_labels.append(f'{rel_video_path} {gt_text}')
        print(f"{rel_video_path} | {gt_text}")
    else:
        break

reduced_label_filepath = os.path.join(speaker_dir, f"train_reduced{max_seconds}_{seed}_labels.txt")
print(f"Total Number of Labels: {len(reduced_labels)}")
print(f"Total Seconds: {total_seconds} | Total Time: {seconds_to_hhmmss(total_seconds)}")

with open(reduced_label_filepath, 'w') as file:
    file.write('\n'.join(reduced_labels))

print(f"Wrote the reduced set of labels to {reduced_label_filepath}")
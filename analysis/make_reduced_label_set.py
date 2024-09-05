# This file is used to create a reduced label set using the existing labels based on the highest WER sentences
import json
import random
import os
import math
import numpy as np
import cv2
import pandas as pd
import copy

seed = 42

data_dir = "/ssd_scratch/cvit/vanshg/datasets/accented_speakers"
speaker_name = "daniel_howell"
speaker_dir = os.path.join(data_dir, f"{speaker_name}")
results_csv_file = '/ssd_scratch/cvit/vanshg/daniel_howel_inference/pretrained_perf_on_all_labels_beam40/lightning_logs/version_0/results/test_results_epoch0.csv'
reduced_label_filepath = os.path.join(speaker_dir, "reduced_labels.txt")

speaker_df = pd.read_csv(results_csv_file)
# Filtering all the rows based on the unique index
speaker_df = speaker_df.drop_duplicates(subset=['Index'])

print(speaker_df)
# speaker_df = speaker_df.sample(frac=1, random_state=seed).reset_index(drop=True) # doing random shuffle instead of MAX WER
speaker_df = speaker_df.sort_values(by='WER', ascending=False)

# List of reduced labels
reduced_labels = []
max_seconds = 3900 # 1 hr 5 mins
total_seconds = 0
total_length = 0
total_word_distance = 0

for index, row in speaker_df.iterrows():
    rel_video_path = row['Video Path']
    video_path = os.path.join(data_dir, rel_video_path)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_in_seconds = total_frames / fps
    if total_seconds + duration_in_seconds <= max_seconds:
        total_seconds += duration_in_seconds
        wer = row['WER']
        sentence_length = row['Length']
        word_distance = row['Word Distance']
        gt_text = row['Ground Truth Text']
        reduced_labels.append(f'{rel_video_path} {gt_text}')

        total_length += sentence_length
        total_word_distance += word_distance
        print(f"{rel_video_path} | WER: {wer} | {gt_text}")

# Randomly shuffling them so there are no problems with anything
rng = random.Random(seed)
rng.shuffle(reduced_labels)

print(f"NUMBER OF REDUCED LABELS: {len(reduced_labels)} | TOTAL LABELS: {len(speaker_df)} | Number Seconds: {total_seconds}")
print(f"Total Length: {total_length} | Total Distance: {total_word_distance} | WER: {total_word_distance/total_length}")
with open(reduced_label_filepath, 'w') as file:
    file.write('\n'.join(reduced_labels))

print(f"Wrote the reduced set of labels to {reduced_label_filepath}")
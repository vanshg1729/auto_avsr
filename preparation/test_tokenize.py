# Script to check if SentencePiece tokenization is enough for GRID dataset
import argparse
import glob
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
import torch
import torchaudio
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

text_transform = TextTransform()
data_dir = "/ssd_scratch/cvit/vanshg/preprocessed_grid"
speaker = "s1"
transcript_dir = os.path.join(data_dir, f"transcription/{speaker}")

filenames = glob.glob(os.path.join(transcript_dir, "*.txt"))
print(len(filenames))
print(filenames[0])

for filename in tqdm(filenames):
    gt_text = open(filename, "r").readline()
    token_id = text_transform.tokenize(gt_text)
    pred_text = text_transform.post_process(torch.tensor(token_id))

    word_dist = compute_word_level_distance(gt_text, pred_text)
    if word_dist == 0:
        print(f"\n{'*' * 70}")
        print(f"GT: {gt_text}")
        print(f"Pred: {pred_text}")

        print(f"dist = {word_dist}")
        print(f"{'*' * 70}")
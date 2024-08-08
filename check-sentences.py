import os
import sys
import torch
import glob
import string

import torchaudio

from datamodule.transforms import TextTransform

transcripts_dir = "/home/vanshg/play/IIITH/research-cvit/lip-reading/auto_avsr/datasets/deaf-youtube/benny-ada/transcriptions"

txt_files = sorted(glob.glob(os.path.join(transcripts_dir, "*.txt")))
text_transform = TextTransform()

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

def process_text(text):
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.upper()
    return text

for i, file_path in enumerate(txt_files):
    text = open(file_path, 'r').read()
    text = process_text(text)

    token_ids = text_transform.tokenize(text)
    transcribed_text = text_transform.post_process(token_ids)

    word_distance = compute_word_level_distance(text, transcribed_text)
    if word_distance:
        print('*' * 70)
        print(f"\n{i} File Path: {file_path}")
        print(f"GT text: {text}")
        print(f"Transcribed text: {transcribed_text}")
        print(f"WD: {word_distance}")
        print(f"Tokens: {token_ids}")


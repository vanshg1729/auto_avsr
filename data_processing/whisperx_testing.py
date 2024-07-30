import os
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import gc
import json

import cv2
import torch
import whisperx

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{device = }")

model_dir = "/ssd_scratch/cvit/vanshg/checkpoints/whisperx"
compute_type = "float16"
model = whisperx.load_model("large-v3", device, compute_type=compute_type, download_root=model_dir, language='en', threads=4)
model_a, metadata = whisperx.load_align_model(language_code='en', device=device, model_dir=model_dir)
print(f"Loaded the model")

video_path = '/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/chess/videos/-3otg-asbvc.mp4'
# video_path = '../data/leah_crop2.mp4'
batch_size = 4

audio = whisperx.load_audio(video_path)
result = model.transcribe(audio, batch_size=batch_size)
print(f"Got the result from model")

segments_file = "./chess_segments.json"
with open(segments_file, 'w') as json_file:
    json.dump(result['segments'], json_file)
    print(f"Wrote segments to {segments_file}")

# del model
# gc.collect()
# torch.cuda.empty_cache()

align_model_dir = '/ssd_scratch/cvit/vanshg/checkpoints'
# model_a, metadata = whisperx.load_align_model(language_code='en', device=device, model_name='WAV2VEC2_ASR_LARGE_LV60K_960H', model_dir=model_dir)
result_align = whisperx.align(result['segments'], model_a, metadata, audio, device, return_char_alignments=False)
print(f"Got the aligned results")

segments_aligned_file = './chess_segments_aligned.json'
with open(segments_aligned_file, 'w') as json_file:
    json.dump(result_align['segments'], json_file)
    print(f"Wrote aligned segments to {segments_aligned_file}")

word_segments_file = './chess_words.json'
with open(word_segments_file, 'w') as json_file:
    json.dump(result_align['word_segments'], json_file)
    print(f"Wrote word segments to {word_segments_file = }")

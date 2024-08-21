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
import sys
import numpy as np

from vtt_utils import read_vtt, process_vtt_entries

# Argument Parsing
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube',
    help="Directory of original dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='jazzy',
    help='Name of speaker'
)

args = parser.parse_args()
src_speaker_dir = os.path.join(args.data_dir, f"{args.speaker}")
src_caption_dir = os.path.join(src_speaker_dir, f"captions")
dst_caption_dir = os.path.join(src_speaker_dir, f"captions")
print(f"SRC Speaker DIR: {src_speaker_dir}")
print(f"SRC Caption DIR: {src_caption_dir}")
print(f"DST Caption DIR: {dst_caption_dir}")

def main():
    caption_files = glob.glob(os.path.join(src_caption_dir, "*.vtt"))
    # caption_files = [os.path.join(src_caption_dir, "vfeDGg2N3sI.en-GB.vtt")]
    print(f"{len(caption_files)}")
    print(f"{caption_files[0] = }")

    for caption_idx, caption_file in enumerate(tqdm(caption_files, desc="Processing Captions")):
        entries = read_vtt(caption_file)
        entries = process_vtt_entries(entries)

        caption_fname = os.path.basename(caption_file).split('.')[0]
        dst_caption_filepath = os.path.join(dst_caption_dir, f"{caption_fname}.json")

        with open(dst_caption_filepath, 'w', encoding='utf-8') as json_file:
            json.dump(entries, json_file, ensure_ascii=False, indent=4)
        
        print(f"Wrote the caption {caption_idx} to {dst_caption_filepath}")

if __name__ == '__main__':
    main()
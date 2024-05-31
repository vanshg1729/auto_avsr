import argparse
import glob
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="LRS2LRS3 Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/gridcorpus',
    help="Directory of original dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="mediapipe",
    help="Type of face detector. (Default: mediapipe)",
)
parser.add_argument(
    "--landmarks-dir",
    type=str,
    default=None,
    help="Directory of landmarks",
)
parser.add_argument(
    "--root-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/preprocessed_grid',
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='s1',
    help='Name of speaker'
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=24,
    help="Max duration (second) for each segment, (Default: 24)",
)
parser.add_argument(
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Merges the audio and video components to a media file.",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel.",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing).",
)
args = parser.parse_args()

seg_duration = args.seg_duration
text_transform = TextTransform()

# Load Data
args.data_dir = os.path.normpath(args.data_dir)
vid_dataloader = AVSRDataLoader(
    modality="video", detector=args.detector, convert_gray=False
)
seg_vid_len = seg_duration * 25

speakers = [1]

# def get_gt_text(src_dir, fname):
#     src_txt_filename = os.path.join(src_dir, f"{fname}.align")
#     words = []
#     with open(src_txt_filename, "r") as file:
#         for line in file:
#             parts = line.split()
#             if len(parts) == 3 and parts[2] != 'sil':
#                 words.append(parts[2])
#         return ' '.join(words).upper()

def get_gt_text(src_dir, fname):
    src_txt_filename = os.path.join(src_dir, f"{fname}.txt")
    f = open(src_txt_filename, "r")
    words = f.readline()
    f.close()
    return words

for speaker in speakers:
    args.speaker = f"s{speaker}"
    print(f"Speaker = {args.speaker}")
    dst_vid_dir = os.path.join(args.root_dir, f"video/{args.speaker}")
    dst_txt_dir = os.path.join(args.root_dir, f"transcription/{args.speaker}")
    print(f"DST vid dir = {dst_vid_dir}")
    print(f"DST txt dir = {dst_txt_dir}")

    dst_label_dir = os.path.join(args.root_dir, f"labels")
    dst_label_file = os.path.join(dst_label_dir, f"{args.speaker}_label.csv")
    os.makedirs(dst_label_dir, exist_ok=True)
    f = open(dst_label_file, "w")

    vid_filenames = glob.glob(os.path.join(dst_vid_dir, "*.mp4"))
    vid_filenames = sorted(vid_filenames)
    print(len(vid_filenames))
    print(vid_filenames[0])

    for data_filename in tqdm(vid_filenames):
        # Reading the preprocessed videos
        try:
            video_data = vid_dataloader.load_video(data_filename)
        except (UnboundLocalError, TypeError, OverflowError, AssertionError):
            continue

        # getting the path of the text file
        fname = os.path.basename(data_filename).split('.')[0]
        dst_txt_filename = os.path.join(dst_txt_dir, f"{fname}.txt")
        gt_text = open(dst_txt_filename, "r").readline()

        # Getting the token string
        token_id_str = " ".join(
            map(str, [_.item() for _ in text_transform.tokenize(gt_text)])
        )
        basename = os.path.basename(data_filename)

        f.write(
            f"{args.speaker},{basename},{video_data.shape[0]},{token_id_str}\n"
        )
        # print(f"saved the data for {fname} to {dst_vid_filename}")

    f.close()
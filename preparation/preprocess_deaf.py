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

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube',
    help="Directory of original dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Type of face detector. (Default: retinaface)",
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
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube',
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='benny',
    help='Name of speaker'
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=24,
    help="Max duration (second) for each segment, (Default: 24)",
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
parser.add_argument(
    '--ngpu',
    type=int,
    default=1,
    help='Number of GPUs to use in Parallel'
)
args = parser.parse_args()

seg_duration = args.seg_duration
text_transform = TextTransform()

# Load Data
args.data_dir = os.path.normpath(args.data_dir)
gpu_id = args.job_index
video_dataloader = AVSRDataLoader(modality="video", detector=args.detector, convert_gray=False, device=f"cuda:{gpu_id}") 
seg_vid_len = seg_duration * 25

def process_text(text):
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.upper()
    return text

def get_gt_text(file_path):
    src_txt_filename = file_path
    with open(src_txt_filename, "r") as file:
        text = ' '.join(file.read().split(' ')[1:])
        text = process_text(text)
    
    return text

src_speaker_dir = os.path.join(args.data_dir, f"{args.speaker}")
src_txt_dir = os.path.join(src_speaker_dir, f"sentence_clips")
src_vid_dir = os.path.join(src_speaker_dir, f"sentence_clips")

dst_speaker_dir = os.path.join(args.root_dir, f"{args.speaker}")
dst_vid_dir = os.path.join(dst_speaker_dir, f"processed_videos")
dst_txt_dir = os.path.join(dst_speaker_dir, f"transcriptions")

print(f"Src video dir = {src_vid_dir}")
print(f"Src txt dir: {src_txt_dir}")
print(f"DST vid dir = {dst_vid_dir}")
print(f"DST txt dir = {dst_txt_dir}")

os.makedirs(dst_vid_dir, exist_ok=True)
os.makedirs(dst_txt_dir, exist_ok=True)

def preprocess_video_file(video_path, args, video_id=0):
    print(f"Processing video {video_id} with path {video_path}")

    vid_folder_name = os.path.basename(os.path.dirname(video_path))
    vid_clips_dir = os.path.join(src_vid_dir, f"{vid_folder_name}")

    video_fname = os.path.basename(video_path).split('.')[0]
    data_filename = os.path.join(vid_clips_dir, f"{video_fname}.mp4")
    try:
        video_data = video_dataloader.load_data(data_filename, None)
        # print(f"shape of video_data = {video_data.shape}")
    except (UnboundLocalError, TypeError, OverflowError, AssertionError):
        return

    dst_vid_filename = os.path.join(dst_vid_dir, f"{video_fname}.mp4")
    dst_txt_filename = os.path.join(dst_txt_dir, f"{video_fname}.txt")

    fname = video_fname.split('.')[0]
    src_txt_filename = os.path.join(vid_clips_dir, f"{video_fname}.txt")
    gt_text = get_gt_text(src_txt_filename)
    save_vid_aud_txt(
        dst_vid_filename,
        None,
        dst_txt_filename,
        video_data,
        None,
        gt_text
    )

    # Getting the token string
    # token_id_str = " ".join(
    #     map(str, [_.item() for _ in text_transform.tokenize(gt_text)])
    # )
    rel_dir = os.path.join(f"{args.speaker}", "processed_videos")
    basename = os.path.basename(dst_vid_filename)
    rel_vid_path = os.path.join(rel_dir, basename)

    print(f"{rel_vid_path} {gt_text}")
    print(f"saved the data for {video_fname} to {dst_vid_filename}")

def main(args):
    # Video Filenames
    vid_filenames = glob.glob(os.path.join(src_vid_dir, "*/*.mp4"))
    vid_filenames = sorted(vid_filenames)

    unit = math.ceil(len(vid_filenames) * 1.0 / args.ngpu)
    vid_filenames = vid_filenames[args.job_index * unit : (args.job_index + 1) * unit]
    print(len(vid_filenames))
    print(vid_filenames[0])

    for i, video_path in enumerate(tqdm(vid_filenames, desc=f"Processing Video")):
        preprocess_video_file(video_path, args, video_id=i)
    
    print(f"PRE-PROCESSED ALL VIDEOS in job {args.job_index}/{args.ngpu} OF SPEAKER {args.speaker}!!!")

if __name__ == '__main__':
    main(args)
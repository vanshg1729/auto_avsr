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

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="TCD-TIMIT preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/TCD-TIMIT/volunteers',
    help="Directory of original dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
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
    default='/ssd_scratch/cvit/vanshg/tcd_processed/volunteers',
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

speakers = ["01M"]

def get_gt_text(file_path):
    src_txt_filename = file_path
    with open(src_txt_filename, "r") as file:
        text = file.read()
        punctuation = string.punctuation.replace("'", "")
        text = text.translate(str.maketrans('', '', punctuation))
        text = text.upper()
    
    return text

def get_gt_text_from_lines(file_path):
    src_txt_filename = file_path
    word_list = []
    with open(src_txt_filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            word = line.split()[-1]
            word_list.append(word)
        
        text = ' '.join(word_list)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.upper()
    
    return text
    
for speaker in speakers:
    args.speaker = f"{speaker}"
    print(f"Speaker = {args.speaker}")
    src_vid_dir = os.path.join(args.data_dir, f"{args.speaker}/straightcam")
    src_txt_dir = os.path.join(args.data_dir, f"{args.speaker}/straightcam")

    dst_data_dir = os.path.join(args.root_dir, f"{args.speaker}/straightcam")
    dst_vid_dir = os.path.join(args.root_dir, f"{args.speaker}/straightcam/videos")
    dst_txt_dir = os.path.join(args.root_dir, f"{args.speaker}/straightcam/transcription")

    print(f"Src video dir = {src_vid_dir}")
    print(f"src txt dir = {src_txt_dir}")
    print(f"DST vid dir = {dst_vid_dir}")
    print(f"DST txt dir = {dst_txt_dir}")

    os.makedirs(dst_data_dir, exist_ok=True)
    os.makedirs(dst_vid_dir, exist_ok=True)
    os.makedirs(dst_txt_dir, exist_ok=True)

    # Label file for speaker
    dst_label_file = os.path.join(dst_data_dir, "labels.txt")
    f = open(dst_label_file, "w")

    vid_filenames = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
    vid_filenames = sorted(vid_filenames)
    print(len(vid_filenames))
    print(vid_filenames[0])

    # Iterating over video files of the speaker
    i = 0
    for data_filename in tqdm(vid_filenames):
        # if i < 296:
        #     i += 1
        #     continue
        # print(i)
        print(f"Processing {i} {data_filename} now")
        try:
            video_data = vid_dataloader.load_data(data_filename, None)
            # print(f"shape of video_data = {video_data.shape}")
        except (UnboundLocalError, TypeError, OverflowError, AssertionError):
            print(f"some error occurred here")
            continue

        fname = os.path.basename(data_filename).split('.')[0]
        src_txt_filename = os.path.join(src_txt_dir, f"{fname.upper()}.txt")
        dst_vid_filename = os.path.join(dst_vid_dir, f"{fname}.mp4")
        dst_txt_filename = os.path.join(dst_txt_dir, f"{fname}.txt")

        gt_text = get_gt_text_from_lines(src_txt_filename)
        print(f"{dst_txt_filename}, {gt_text}")
        save_vid_aud_txt(
            dst_vid_filename,
            None,
            dst_txt_filename,
            video_data,
            None,
            gt_text
        )

        # # Getting the token string
        # token_id_str = " ".join(
        #     map(str, [_.item() for _ in text_transform.tokenize(gt_text)])
        # )
        basename = f"speaker{args.speaker}/straightcam/{fname}.mp4"
        # basename = os.path.basename(dst_vid_filename)

        f.write(
            f"{basename} {gt_text}\n"
        )
        print(f"saved the data for {fname} to {dst_vid_filename}")
        i += 1
    f.close()
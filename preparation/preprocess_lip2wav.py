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
    default='/ssd_scratch/cvit/vanshg/datasets/accented_speakers',
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
    '--speaker',
    type=str,
    default='crazy_russian',
    help='Name of speaker'
)
parser.add_argument(
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Merges the audio and video components to a media file.",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=24,
    help="Max duration (second) for each segment, (Default: 24)",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=3,
    help="Index to identify separate jobs (useful for parallel processing).",
)
parser.add_argument(
    '--ngpu',
    type=int,
    default=4,
    help='Number of GPUs to use in Parallel'
)
args = parser.parse_args()

seg_duration = args.seg_duration
text_transform = TextTransform()

# Load Data
args.data_dir = os.path.normpath(args.data_dir)
gpu_id = args.job_index
video_dataloader = AVSRDataLoader(modality="video", detector=args.detector, 
                                  convert_gray=False, device=f"cuda:{gpu_id}") 
audio_dataloader = AVSRDataLoader(modality='audio')
seg_vid_len = seg_duration * 25

def process_text(text):
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))
    text = ' '.join(text.split()) # Removing extra whitespaces
    text = text.upper()
    return text

def get_gt_text(file_path):
    src_txt_filename = file_path
    with open(src_txt_filename, "r", encoding='utf-8') as file:
        text = ' '.join(file.read().split(' ')[1:])
        text = process_text(text)
    
    return text

src_speaker_dir = os.path.join(args.data_dir, f"{args.speaker}")
src_txt_dir = os.path.join(src_speaker_dir, f"sentence_clips")
src_vid_dir = os.path.join(src_speaker_dir, f"sentence_clips")

dst_speaker_dir = os.path.join(args.data_dir, f"{args.speaker}")
dst_vid_dir = os.path.join(dst_speaker_dir, f"processed_videos")
dst_aud_dir = os.path.join(dst_speaker_dir, f"processed_audio")
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
        audio_data = audio_dataloader.load_data(data_filename)
        # print(f"shape of video_data = {video_data.shape}")
    except (UnboundLocalError, TypeError, OverflowError, AssertionError):
        return

    src_txt_filename = os.path.join(vid_clips_dir, f"{video_fname}.txt")
    gt_text = get_gt_text(src_txt_filename)
    print(f"{gt_text = }")

    dst_vid_filename = os.path.join(dst_vid_dir, f"{video_fname}.mp4")
    dst_txt_filename = os.path.join(dst_txt_dir, f"{video_fname}.txt")
    dst_aud_filename = os.path.join(dst_aud_dir, f"{video_fname}.wav")

    save_vid_aud_txt(
        dst_vid_filename,
        dst_aud_filename,
        dst_txt_filename,
        video_data,
        audio_data,
        gt_text,
        video_fps=25,
        audio_sample_rate=16000,
    )

    if args.combine_av:
        in1 = ffmpeg.input(dst_vid_filename)
        in2 = ffmpeg.input(dst_aud_filename)
        out = ffmpeg.output(
            in1["v"],
            in2["a"],
            dst_vid_filename[:-4] + ".av.mp4",
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            loglevel="panic",
        )
        out.run()
        shutil.move(dst_vid_filename[:-4] + ".av.mp4", dst_vid_filename)

    # Relative path starting from the root of the dataset
    basename = os.path.relpath(dst_vid_filename, 
                               start=args.data_dir)

    print(f"{basename} {gt_text}")
    print(f"saved the data for {video_fname} to {dst_vid_filename}")

def main(args):
    # Video Filenames
    # video_ids_file = os.path.join(src_speaker_dir, "all_video_ids.txt")
    # video_ids = open(video_ids_file, 'r').read().split()
    # vid_filenames = []
    # print(f"{video_ids = }")
    # for video_id in video_ids:
    #     vid_filenames += glob.glob(os.path.join(src_vid_dir, f"{video_id}/*.mp4"))
    # print(len(vid_filenames))
    # vid_filenames = sorted(vid_filenames)

    vid_filenames = glob.glob(os.path.join(src_vid_dir, "*/*.mp4"))
    vid_filenames = sorted(vid_filenames)
    # vid_filenames = glob.glob(os.path.join(src_vid_dir, "EiEIfBatnH8_crop/*.mp4"))
    print(f"Total number of Video Files: {len(vid_filenames)}")
    print(f"{vid_filenames[0] = }")

    unit = math.ceil(len(vid_filenames) * 1.0 / args.ngpu)
    vid_filenames = vid_filenames[args.job_index * unit : (args.job_index + 1) * unit]
    print(f"{len(vid_filenames) = }")
    print(vid_filenames[0])
    print(f"Number of files for this job index: {len(vid_filenames)}")

    for i, video_path in enumerate(tqdm(vid_filenames, desc=f"Processing Video")):
        preprocess_video_file(video_path, args, video_id=i)
    
    print(f"PRE-PROCESSED ALL VIDEOS in job {args.job_index} OF SPEAKER {args.speaker}!!!")

if __name__ == '__main__':
    main(args)
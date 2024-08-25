import os
import sys
import glob
import time
import json
from collections import defaultdict

from tqdm import tqdm
import cv2
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='./datasets/deaf-youtube',
    help="Directory of original dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='mia_sandra',
    help='Name of speaker'
)
parser.add_argument(
    '--num-workers',
    help='Number of processes (jobs) across which to run in parallel',
    default=36,
    type=int
)
args = parser.parse_args()

src_speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(src_speaker_dir, "videos")
src_ocr_dir = os.path.join(src_speaker_dir, "ocr_outputs")
src_crops_dir = os.path.join(src_speaker_dir, "gaussian_bbox_frames")

dst_transcript_dir = os.path.join(src_speaker_dir, "ocr_transcripts")

def process_video_file(video_path):
    print(f"Processing video: {video_path}")
    video_name = os.path.basename(video_path).split('.')[0]
    src_vid_ocr_dir = os.path.join(src_ocr_dir, f"{video_name}")
    dst_vid_transcript_dir = os.path.join(dst_transcript_dir, f"{video_name}")
    src_vid_crops_dir = os.path.join(src_crops_dir, f"{video_name}")
    dst_ocr_jsonpath = os.path.join(dst_vid_transcript_dir, "ocr_outputs.json")
    dst_transcript_json = os.path.join(dst_vid_transcript_dir, f"{video_name}.json")
    ocr_outputs = []

    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(dst_vid_transcript_dir, exist_ok=True)

    ocr_text_files = sorted(glob.glob(os.path.join(src_vid_ocr_dir, "*.txt")))
    for txt_file in ocr_text_files:
        with open(txt_file, 'r') as file:
            line = file.read()
            text = ' '.join(line.split(' ')[:-1])
            confidence = float(line.split(' ')[-1])
            if text == 'None':
                text = None

        frame_name = os.path.basename(txt_file).split('.')[0]
        frame_idx = frame_name.split('_')[1]
        frame_filepath = os.path.join(src_vid_crops_dir, f"{frame_name}.png")

        frame_ocr_out = {
            'frame_number': int(frame_idx),
            'frame_idx': frame_idx,
            'frame_filepath': frame_filepath,
            'ocr_text_file': txt_file,
            'ocr_text': text,
            'confidence': confidence
        }

        # print(frame_ocr_out)
        ocr_outputs.append(frame_ocr_out)
    
    ocr_json = {
        'video_path': video_path,
        'fps': fps,
        'total_frames': total_frames,
        'ocr_outputs': ocr_outputs
    }
    with open(dst_ocr_jsonpath, 'w') as jsonfile:
        json.dump(ocr_json, jsonfile)
    
    ocr_transcripts = get_video_ocr_transcripts(ocr_outputs, fps)
    with open(dst_transcript_json, 'w') as jsonfile:
        json.dump(ocr_transcripts, jsonfile)

def get_video_ocr_transcripts(frame_ocr_outputs, fps):
    active_transcript = False
    cur_text = ""
    st_frame = -1
    end_frame = -1
    text_freq = defaultdict(int)

    transcripts = []

    for frame_output in frame_ocr_outputs:
        frame_text = frame_output['ocr_text']
        frame_number = frame_output['frame_number']
        if active_transcript:
            if frame_text:
                end_frame = frame_number
                text_freq[frame_text] += 1
            else:
                cur_freq = 0
                for text, freq in text_freq.items():
                    if freq > cur_freq:
                        cur_text = text
                        cur_freq = freq

                transcript = {
                    'start': st_frame/fps,
                    'end': end_frame/fps,
                    'start_frame': st_frame,
                    'end_frame': end_frame,
                    'text': cur_text,
                    'text_frequency': cur_freq,
                    'text_length': len(cur_text.split())
                }
                transcripts.append(transcript)

                active_transcript = False
                text_freq = defaultdict(int)
                st_frame = -1
                end_frame = -1
                cur_text = ""
        elif frame_text:
            active_transcript = True
            st_frame = frame_number
            end_frame = frame_number
            text_freq[frame_text] += 1
    
    if active_transcript:
        cur_freq = 0
        for text, freq in text_freq.items():
            if freq > cur_freq:
                cur_text = text
                cur_freq = freq

        transcript = {
            'start': st_frame/fps,
            'end': end_frame/fps,
            'start_frame': st_frame,
            'end_frame': end_frame,
            'text': cur_text,
            'text_frequency': cur_freq,
            'text_length': len(cur_text.split())
        }
        transcripts.append(transcript)
    
    return transcripts

def main(args):
    vid_filenames = [os.path.join(src_vid_dir, "qAt94Wmcavw.mkv")]

    for i, video_path in enumerate(tqdm(vid_filenames, desc=f"Processing Video")):
        process_video_file(video_path)

if __name__ == '__main__':
    main(args)
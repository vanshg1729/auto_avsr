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

import cv2
import numpy as np
import torch

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback, subprocess

from video_utils import clip_video_ffmpeg, video_frame_batch_generator
from track_utils import *
from face_tracker import FaceTracker

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument Parsing
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='../datasets/Lip2Wav',
    help="Directory of original dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="yolov5",
    choices=['retinaface', 'yolov5', 'face_alignment'],
    help="Type of face detector. (Default: face_alignment)",
)
parser.add_argument(
    "--root-dir",
    type=str,
    default='../datasets/Lip2Wav',
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='chem',
    help='Name of speaker'
)
parser.add_argument(
    '--ngpu',
    help='Number of GPUs across which to run in parallel',
    default=1,
    type=int
)
parser.add_argument(
    '--batch-size',
    help='Single GPU Face Detection batch size',
    default=2,
    type=int
)

args = parser.parse_args()
print(f"Detecting faces using : {args.detector}")

src_speaker_dir = os.path.join(args.data_dir, args.speaker)
src_vid_dir = os.path.join(src_speaker_dir, "videos")
dst_vid_dir = os.path.join(src_speaker_dir, "face_tracks_test")
print(f"Src video dir = {src_vid_dir}")
print(f"DST vid dir = {dst_vid_dir}")

video_files = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
video_files = sorted(video_files)
print(f"Total number of Video Files: {len(video_files)}")
print(f"{video_files[0] = }")

if args.detector == 'retinaface':
    from ibug.face_alignment import FANPredictor
    from ibug.face_detection import RetinaFacePredictor
    model_name = "resnet50"
    face_detectors = [RetinaFacePredictor(device=f"cuda:{i}", threshold=0.8,
                                        model=RetinaFacePredictor.get_model(model_name))
                                        for i in range(args.ngpu)]
elif args.detector == 'yolov5':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from preparation.detectors.yoloface.face_detector import YoloDetector
    face_detectors = [YoloDetector(device=f"cuda:{i}", min_face=25)
                      for i in range(args.ngpu)]
elif args.detector == 'face_alignment':
    # Face Alignment Detector
    import face_detection
    from face_detection import FaceAlignment
    face_detectors = [FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                    device=f"cuda:{i}") for i in range(args.ngpu)]
else:
     raise ValueError(f"'{args.detector = }' Detector is not a valid choice for face detector")

def save_track(video_path, track, output_path, fps):
    start_frame = track['start_frame']
    end_frame = track['end_frame']
    num_frames = end_frame - start_frame + 1

    start_time = start_frame/fps
    end_time = end_frame/fps
    timestamp = (start_time, end_time)

    # Don't save the video if it is less than 1 second
    if num_frames < fps:
        print(f"\nvideo track is less than 1 second: {num_frames = } | {start_frame = } | {end_frame = }")
        return {}

    print(f"\nStart Frame: {start_frame} | End Frame: {end_frame}")
    clip_video_ffmpeg(video_path, timestamp, output_path, verbose=True)
    track_metadata = {'input_path': video_path, 'output_path': output_path,
                      'start_time': start_time, 'end_time': end_time, 'fps': fps,
                      "start_frame": start_frame, "end_frame": end_frame}
    print(f"Saved the face track with {num_frames = } to {output_path}")

    return track_metadata

def crop_frame(frame, speaker):
	if speaker == "chem" or speaker == "hs" or speaker == "dl":
		return frame
	elif speaker == "chess":
		H, W = frame.shape[:2]
		return frame[H//3:, W//2:]
	elif speaker == "dl" or speaker == "eh":
		return  frame[int(frame.shape[0]*3/4):, int(frame.shape[1]*3/4): ]
	else:
		raise ValueError("Unknown speaker!")
		exit()

def process_video_file(video_path, args, gpu_id=0, video_id=0):
    print(f"Processing video: {video_path}")
    face_detector = face_detectors[gpu_id]
    
    # Getting the output clips directory for this video
    video_fname = os.path.basename(video_path).split('.')[0]
    clips_dir = os.path.join(dst_vid_dir, f"{video_fname}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get the total number of frames and fps
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Number of Frames: {total_frames} | FPS: {fps}")
    batch_size = args.batch_size

    # FACE TRACKING
    face_tracker = FaceTracker(iou_threshold=0.5, min_frames_thresh=fps, verbose=True)
    tracks_metadata = []
    metadata_filepath = os.path.join(clips_dir, f"tracks.json")
    video_loader = video_frame_batch_generator(video_path, batch_size)

    # Batch Processing of Frames
    for frame_ids, frames in tqdm(video_loader, total=total_frames//batch_size, 
                                  desc=f"Processing video {video_id} Frame Batches"):
        if args.detector == 'retinaface':
            preds = get_batch_prediction_retinaface(frames, face_detector)
            frames_bboxes = get_bboxes_from_retina_preds(preds)
        elif args.detector == 'yolov5':
             frames_bboxes = get_batch_prediction_yolov5(frames, face_detector)

        # process each frame individually to get face tracks
        for i in range(len(frames)):
            # Get the frame and detections for this frame
            frame, frame_idx = frames[i], frame_ids[i]
            detections = frames_bboxes[i]
            
            # Update the face tracker
            face_tracker.update(detections, frame_idx)

            # Update the tracker again in case of last frame
            if frame_idx == total_frames - 1:
                 face_tracker.update([], frame_idx + 1)

            # Look through all the saved tracks to find any new tracks
            for track in face_tracker.saved_tracks:
                 if track['saved'] == False:
                    save_id = track['save_id']
                    out_vid_path = os.path.join(clips_dir, f"track-{save_id}.mp4")
                    track_metadata = save_track(video_path, track, out_vid_path, fps)
                    if len(track_metadata):
                         tracks_metadata.append(track_metadata)
                    
                    track['saved'] = True

    # Save the tracks.json metadata file after all tracks have been detected
    metadata_file_dir = os.path.dirname(metadata_filepath)
    os.makedirs(metadata_file_dir, exist_ok=True)
    with open(metadata_filepath, 'w') as json_file:
        json.dump(tracks_metadata, json_file, indent=4)
        print(f"Saved the tracks metadata to {metadata_filepath}")

def mp_handler(job):
    video_path, args, gpu_id, video_id = job
    try:
        process_video_file(video_path, args, gpu_id, video_id)
    except KeyboardInterrupt:
        exit(0)

def main(args):
    video_files = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
    video_files = sorted(video_files)
    print(f"Total number of Video Files: {len(video_files)}")
    print(f"{video_files[0] = }")

    jobs = [(video_path, args, i % args.ngpu, i) for i, video_path in enumerate(video_files)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    print(f"WROTE FACE_TRACKS OF ALL VIDEOS OF SPEAKER {args.speaker} !!!")

if __name__ == '__main__':
    main(args)
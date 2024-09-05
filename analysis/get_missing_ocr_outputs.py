import os
import sys
import glob
import time

from paddleocr import PaddleOCR

from tqdm import tqdm
import cv2
import numpy as np

import argparse
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
    default='mia_sandra',
    help='Name of speaker'
)
args = parser.parse_args()

src_speaker_dir = os.path.join(args.data_dir, args.speaker)

src_vid_dir = os.path.join(src_speaker_dir, "videos")
src_crops_dir = os.path.join(src_speaker_dir, "gaussian_bbox_frames")
dst_ocr_dir = os.path.join(src_speaker_dir, "ocr_outputs")

print(f"SRC SPEAKER DIR: {src_speaker_dir}, {os.path.exists(src_speaker_dir)}")
print(f"SRC VIDEO DIR: {src_vid_dir}, {os.path.exists(src_vid_dir)}")
print(f"SRC CROPS DIR: {src_crops_dir}, {os.path.exists(dst_ocr_dir)}")
print(f"DST OCR DIR: {dst_ocr_dir}")

ocr = PaddleOCR(lang='en')
def process_frame(ocr, frame_path, frame_idx, video_path, output_dir):
    padding = 5
    frame_name = f"frame_{str(frame_idx).zfill(padding)}"
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"Processing frame {frame_idx} of video {video_name}")

    dst_vid_ocr_dir = os.path.join(output_dir, video_name)
    os.makedirs(dst_vid_ocr_dir, exist_ok=True)

    output_file = os.path.join(dst_vid_ocr_dir, f"{frame_name}.txt")

    results = ocr.ocr(frame_path, cls=True)
    confidence = 1.0
    if results and results[0]:
        transcribed_text = ' '.join([result[1][0] for result in results[0]])
        confidence = min([result[1][1] for result in results[0]])
    else:
        transcribed_text = None

    with open(output_file, 'w') as f:
        f.write(f"{transcribed_text} {confidence}")
    
    return transcribed_text

def main(args):
    # For 720p
    x, y, w, h = 5, 582, 1266, 135

    video_ids_file = os.path.join(src_speaker_dir, "new_video_ids2.txt")
    video_ids = open(video_ids_file, 'r').read().split()
    print(f"{video_ids = }")
    video_files = [os.path.join(src_vid_dir, f"{video_id}.mkv") for video_id in video_ids]

    # video_files = glob.glob(os.path.join(src_vid_dir, f"*.mkv"))
    video_files = sorted(video_files)
    print(f"Number of video_files: {len(video_files)}")

    for video_path in video_files:
        # Get the total number of frames to calculate padding
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"Total Number of Frames: {total_frames}")
        video_name = os.path.basename(video_path).split('.')[0]
        src_vid_crops_dir = os.path.join(src_crops_dir, f"{video_name}")
        dst_vid_ocr_dir = os.path.join(dst_ocr_dir, f"{video_name}")

        print(f"\n{'*' * 70}")
        print(f"Missing OCR Ouputs for {video_path = }")
        for i in range(total_frames):
            frame_path = os.path.join(src_vid_crops_dir, f"frame_{i:05d}.png")
            ocr_text_path = os.path.join(dst_vid_ocr_dir, f"frame_{i:05d}.txt")

            if not os.path.exists(frame_path):
                print(f"{frame_path} does not exists")
                cap = cv2.VideoCapture(video_path)
                # Set the video position to the desired frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    print(f"Not able to read the frame: {frame_path = }")
                    
                cap.release()

                if frame is not None:
                    # Cropping and procssing the frame
                    cropped_frame = frame[y:y+h, x:x+w]
                    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                    white_parts = np.copy(gray_frame)
                    white_parts[gray_frame >= 240] = 255
                    white_parts[gray_frame < 240] = 0

                    # Writing the file again
                    frame_filename = os.path.join(src_vid_crops_dir, f"frame_{i:05d}.png")
                    cv2.imwrite(frame_filename, white_parts)

            if not os.path.exists(ocr_text_path):
                print(f"{ocr_text_path = } | {os.path.exists(ocr_text_path)}")
                print(f"{frame_path = } | {os.path.exists(frame_path)}\n")

                if os.path.exists(frame_path):
                    process_frame(ocr, frame_path, i, video_path, dst_ocr_dir)
        print(f"{'*' * 70}")

if __name__ == '__main__':
    main(args)
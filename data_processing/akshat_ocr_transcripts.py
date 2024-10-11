import os
import glob
import time

from multiprocessing import Pool, cpu_count

print(f"Right before importing paddleocr: {time.time()}")
from paddleocr import PaddleOCR
import numpy as np
print(f"Right after importing paddleocr: {time.time()}")

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
parser.add_argument(
    '--num-workers',
    help='Number of processes (jobs) across which to run in parallel',
    default=36,
    type=int
)
args = parser.parse_args()

src_speaker_dir = os.path.join(args.data_dir, args.speaker)

src_vid_dir = os.path.join(src_speaker_dir, "raw_videos")
src_crops_dir = os.path.join(src_speaker_dir, "gaussian_bbox_frames")
dst_ocr_dir = os.path.join(src_speaker_dir, "ocr_testing_akshat")

video_name = "3aAi2dqQ1iI"
src_vid_crops_dir = os.path.join(src_crops_dir, f"{video_name}")
dst_vid_ocr_dir = os.path.join(dst_ocr_dir, f"{video_name}")
print(f"SRC VID CROPS DIR: {src_vid_crops_dir}")
print(f"DST VID OCR DIR: {dst_vid_ocr_dir}")
os.makedirs(dst_vid_ocr_dir, exist_ok=True)

# Get a sorted list of all image files in the folder
frame_filepaths = glob.glob(os.path.join(src_vid_crops_dir, "*.jpg"))
frame_filepaths = sorted(frame_filepaths)
print(f"Number of Frame Filepaths: {len(frame_filepaths)}")

# Initialize PaddleOCR with English language support
ocr = PaddleOCR(lang='en')  # Download and load the model into memory

def process_image(image_file):
    # image_path = os.path.join(src_vid_dir, image_file)

    # Perform OCR on the current image
    results = ocr.ocr(image_file, cls=True)

    # Prepare the result string
    output = f"\nProcessing {image_file}"

    # Check if any text was detected
    if results[0]:
        for result in results[0]:  # Access the first list within results
            text = result[1][0]    # Extract the text
            confidence = result[1][1]  # Extract the confidence score
            output += f"\nDetected Text: {text}\nConfidence: {confidence}"
    else:
        output += "\nNo text detected."
    
    # Extract frame number from the file name
    frame_no = os.path.splitext(image_file)[0].split('_')[-1]
    
    # Path for the .nice file
    nice_file_path = os.path.join(dst_vid_ocr_dir, f"{frame_no}.txt")
    
    # Write the OCR results to the .nice file
    with open(nice_file_path, 'w') as file:
        file.write(output)

    return output

def main():
    # Determine the number of CPU cores to use
    # num_cpus = cpu_count()
    num_cpus = args.num_workers

    # Create a pool of workers
    with Pool(num_cpus) as pool:
        # Map the process_image function to the image files
        results = pool.map(process_image, frame_filepaths)

    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
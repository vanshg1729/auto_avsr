import math
import argparse
import subprocess
import sys
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Downloading Videos")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/',
    help="Directory of original dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='jazzy',
    help='Name of speaker'
)
parser.add_argument(
    '--num-jobs',
    help='Number of processes (jobs) across which to run in parallel',
    default=1,
    type=int
)
parser.add_argument(
    '--job-index',
    type=int,
    default=0,
    help='Index to identify separate jobs (useful for parallel processing)'
)
args = parser.parse_args()

data_dir = "/ssd_scratch/cvit/vanshg/datasets/deaf-youtube"
speaker_name = "jazzy"

def download_video(video_id, download_dir):
   # Ensure the download path exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Construct the youtube-dl command
    command = [
        "youtube-dl",
        "-f", "best",
        "https://www.youtube.com/watch?v=" + video_id,
        "-o", os.path.join(download_dir, "%(id)s.%(ext)s")
    ]

    # Run the command and stream output and errors in real-time
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        for stderr_line in iter(process.stderr.readline, ""):
            print(stderr_line, end="", file=sys.stderr)

        process.stdout.close()
        process.stderr.close()
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("Process terminated by user.")

def download_caption(video_id, download_dir, language_code='en'):
    # Ensure the download path exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Construct the youtube-dl command for captions
    command = [
        "youtube-dl",
        "--write-sub",  # Write subtitle file
        "--skip-download",  # Skip downloading the video
        # "--all-subs",
        "--sub-format", "vtt",  # Get subtitles in VTT format
        # "--sub-lang", language_code,  # Specify subtitle language
        "https://www.youtube.com/watch?v=" + video_id,  # Video URL
        "-o", os.path.join(download_dir, "%(id)s.%(ext)s")
    ]

    # Run the command and stream output and errors in real-time
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        for stderr_line in iter(process.stderr.readline, ""):
            print(stderr_line, end="", file=sys.stderr)

        process.stdout.close()
        process.stderr.close()
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("Process terminated by user.")

def main(args):
    speaker_dir = os.path.join(args.data_dir, f"{args.speaker}")
    video_ids_file = os.path.join(speaker_dir, f"all_video_ids.txt") # Path to text file with video ids
    dst_vid_dir = os.path.join(speaker_dir, f"videos") # Path where videos will be downloaded
    dst_caption_dir = os.path.join(speaker_dir, f"captions")

    video_ids = []
    with open(video_ids_file, 'r') as file:
        for line in file:
            video_id = line.strip()
            if video_id:
                video_ids.append(video_id)

    video_ids = video_ids[25:]
    unit = math.ceil(len(video_ids) * 1.0/args.num_jobs)
    video_ids = video_ids[args.job_index * unit : (args.job_index + 1) * unit]
    print(f"Number of video ids for this job index {args.job_index}: {len(video_ids)}")

    print(f"{len(video_ids) = }")
    for i, video_id in enumerate(tqdm(video_ids, desc="Downloading Videos")):
        # if i > 0:
        #     break
        download_video(video_id, dst_vid_dir)
        download_caption(video_id, dst_caption_dir)

if __name__ == "__main__":
    main(args)
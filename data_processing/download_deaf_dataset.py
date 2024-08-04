import subprocess
import sys
import os

from tqdm import tqdm

data_dir = "/ssd_scratch/cvit/vanshg/datasets/deaf-youtube"
speaker_name = "benny-large"

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

if __name__ == "__main__":
    speaker_dir = os.path.join(data_dir, f"{speaker_name}")
    video_ids_file = os.path.join(speaker_dir, f"new_videos.txt") # Path to text file with video ids
    dst_vid_dir = os.path.join(speaker_dir, f"videos") # Path where videos will be downloaded
    dst_caption_dir = os.path.join(speaker_dir, f"captions")

    video_ids = []
    with open(video_ids_file, 'r') as file:
        for line in file:
            video_id = line.strip()
            if video_id:
                video_ids.append(video_id)

    print(f"{len(video_ids) = }")
    for i, video_id in enumerate(tqdm(video_ids, desc="Downloading Videos")):
        # if i > 0:
        #     break
        download_video(video_id, dst_vid_dir)
        download_caption(video_id, dst_caption_dir)
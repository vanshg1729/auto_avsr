import subprocess
import os

def split_video(input_file, output_dir, clip_length=1200):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the video name without extension
    video_name = os.path.basename(input_file).split('.')[0]

    # Command to split the video
    command = [
        'ffmpeg',                  # Call the ffmpeg executable
        '-i', input_file,          # Specify the input video file
        '-map', '0',               # Map all streams from the input file
        # '-c', 'copy',
        '-c:v', 'libx264',          
        '-c:a', 'aac',
        '-segment_time', str(clip_length),  # Set the segment length (in seconds)
        '-f', 'segment',           # Instruct ffmpeg to output segmented files
        '-reset_timestamps', '1',  # Reset timestamps at the start of each segment
        os.path.join(output_dir, f'{video_name}_%03d.mp4')  # Output file pattern with video name and split ID
    ]

    # Run the command
    subprocess.run(command, check=True)

# Example usage
input_video = '/ssd_scratch/cvit/vanshg/datasets/accented_speakers/jack/raw_videos/ETWbMaxGmbA.mp4'
output_directory = '/ssd_scratch/cvit/vanshg/datasets/accented_speakers/jack/videos'
os.makedirs(output_directory)
split_video(input_video, output_directory)
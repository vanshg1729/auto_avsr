import os
import subprocess

data_dir = "/ssd_scratch/cvit/vanshg/datasets/accented_speakers"
speaker = "diane_jennings"
speaker_dir = os.path.join(data_dir, speaker)
src_vid_dir = os.path.join(speaker_dir, "videos")

def get_video_resolution(video_path):
    # Use ffprobe to get video resolution
    command = [
        'ffprobe', 
        '-v', 'error',  # Suppress unnecessary output
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', 
        '-of', 'csv=s=x:p=0',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        resolution = result.stdout.strip()
        if not resolution:
            resolution = "Resolution not found"
        return resolution
    except Exception as e:
        return f"Error: {str(e)}"

def print_video_resolutions(folder_path, output_file):
    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more extensions if needed
                video_path = os.path.join(folder_path, filename)
                resolution = get_video_resolution(video_path)
                f.write(f"{filename}: {resolution}\n")
                print(f"{filename}: {resolution}")

if __name__ == "__main__":
    folder_path = src_vid_dir  # Replace with your folder path
    output_file = os.path.join(src_vid_dir, "video_resolutions.txt")  # Output file to store the resolutions
    print_video_resolutions(folder_path, output_file)
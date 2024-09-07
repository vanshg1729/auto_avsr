import os
import subprocess
import yt_dlp

data_dir = "/ssd_scratch/cvit/vanshg/datasets/lip2wav"
speaker = "chess"
speaker_dir = os.path.join(data_dir, speaker)
video_ids_file = os.path.join(speaker_dir, "train.txt")

def get_video_lengths(video_ids, output_file):
    ydl_opts = {
        'quiet': True,  # Suppress output except for errors
        'skip_download': True,  # We only want metadata, not the video
        'force_generic_extractor': False  # Ensure we use the YouTube extractor
    }

    with open(output_file, 'w') as f:
        for video_id in video_ids:
            url = f"https://www.youtube.com/watch?v={video_id}"
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=False)
                    duration = info_dict.get('duration', 'N/A')  # Duration in seconds
                    
                    # Calculate hours, minutes, and seconds
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60

                    # Format duration
                    duration_str = f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"
                    print(f"{video_id}: {duration_str}\n")
                    f.write(f"{video_id}: {duration_str}\n")
            except Exception as e:
                f.write(f"{video_id}: Error fetching video duration - {str(e)}\n")

if __name__ == "__main__":
    video_ids = open(video_ids_file).read().split()
    output_file = os.path.join(speaker_dir, "train_video_lengths.txt")
    get_video_lengths(video_ids, output_file)
    print(f"Video lengths written to {output_file}")

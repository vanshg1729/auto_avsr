import os
import subprocess

def seconds_to_hhmmss(seconds):
    """
    Convert seconds to hh:mm:ss format.
    
    Args:
        seconds (float): Time in seconds.
    
    Returns:
        str: Time in hh:mm:ss format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def clip_video_ffmpeg(video_path, timestamsp, output_path):
    output_dir = os.path.dirname(output_path)

    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    start_time, end_time = timestamsp
    start_time = seconds_to_hhmmss(start_time)
    end_time = seconds_to_hhmmss(end_time)
    print(f"{start_time = } | {end_time = }")

    # Construct the ffmpeg command
    command = [
        'ffmpeg',
        '-loglevel', 'panic',           # suppress output except for fatal errors
        '-y',                            # Overwrite output file if it exists
        '-ss', start_time,          # Start time
        '-to', end_time,            # End time
        '-i', video_path,                # Input file
        '-c', 'copy',                    # Copy video and audio codec
        # '-reset_timestamps', '1',       # Avoid negative timestamps
        # '-avoid_negative_ts', 'make_zero', # Avoid negative timestamps
        output_path                    # Output file
    ]

    # Run the ffmpeg command
    subprocess.run(command, check=True)
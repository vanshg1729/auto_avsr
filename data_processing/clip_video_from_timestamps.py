import subprocess
import os

input_file = "/ssd_scratch/cvit/akshat/datasets/accented_speakers/aussie_english/videos/Xq0wIzwobOg.mkv"
segments = [
    ("00:00:00", "00:02:30"),
    ("00:14:24", "01:17:00"),
]

# Extract base name without extension
base_name = os.path.splitext(os.path.basename(input_file))[0]
print(f"Base Name: {base_name}")

for i, (start, end) in enumerate(segments, start=1):
    output_file = f"{base_name}_{i}.mkv"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", start,
        "-to", end,
        "-i", input_file,
        "-c", "copy",
        output_file
    ]
    print(f"Clipping {start} to {end} -> {output_file}")
    subprocess.run(cmd, check=True)

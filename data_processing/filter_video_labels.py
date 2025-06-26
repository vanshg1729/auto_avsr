from pathlib import Path

# Paths to your files
label_file_path = "/ssd_scratch/cvit/akshat/datasets/accented_speakers/supersymo/reduced_labels.txt"
remove_file_path = "/ssd_scratch/cvit/akshat/datasets/accented_speakers/supersymo/supersymo_remove_clips.txt"
output_file_path = "/ssd_scratch/cvit/akshat/datasets/accented_speakers/supersymo/reduced_labels_filtered.txt"

# Read the video names to remove (just the base name of the path)
with open(remove_file_path, 'r') as f:
    remove_names = set(Path(line.strip()).name for line in f)

# Filter the label file
with open(label_file_path, 'r') as f:
    lines = f.readlines()

filtered_lines = []
for line in lines:
    video_path = line.split()[0]  # First part is the video path
    video_name = Path(video_path).name
    if video_name not in remove_names:
        filtered_lines.append(line)

# Save filtered output
with open(output_file_path, 'w') as f:
    f.writelines(filtered_lines)

print(f"Filtered labels saved to: {output_file_path}")

import os
import cv2
from tqdm import tqdm
import glob

data_dir = "/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset"
speaker_name = "dl"

src_speaker_dir = os.path.join(data_dir, f"{speaker_name}")
src_video_dir = os.path.join(src_speaker_dir, "processed_videos")
src_text_dir = os.path.join(src_speaker_dir, f"transcriptions")
label_filepath = os.path.join(src_speaker_dir, "labels.txt")

print(f"SRC Speaker DIR: {src_speaker_dir}, {os.path.exists(src_speaker_dir)}")
print(f"SRC Video DIR: {src_video_dir}, {os.path.exists(src_video_dir)}")
print(f"SRC Text DIR: {src_text_dir}, {os.path.exists(src_text_dir)}")

f = open(label_filepath, 'r')
video_list = f.readlines()
max_frames = 0
max_idx = -1
max_frame_path = ""

for idx, line in enumerate(tqdm(video_list)):
    video_metadata = video_list[idx]
    rel_path, gt_text = video_metadata.split()[0], " ".join(video_metadata.split()[1:])

    video_filepath = os.path.join(data_dir, rel_path)
    cap = cv2.VideoCapture(video_filepath)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if total_frames > max_frames:
        max_frames = total_frames
        max_idx = idx
        max_frame_path = video_filepath

print(f"Max frames {max_idx} path = {max_frame_path} | {max_frames = }")

# video_files = glob.glob(os.path.join(src_video_dir, "*.mp4"))
# video_files = sorted(video_files)
# print(f"{len(video_files) = }")
# print(f"{video_files[0] = }")

# for video_idx, video_file in enumerate(tqdm(video_files, desc="Creating Labels for videos")):
#     cap = cv2.VideoCapture(video_file)
#     total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     cap.release()
    
#     video_fname = os.path.basename(video_file).split('.')[0]

#     dst_txt_filename = os.path.join(src_text_dir, f"{video_fname}.txt")

#     if total_frames > 500 or total_frames <= 0:
#         continue
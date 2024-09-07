import os
import shutil
import sys
import glob
from tqdm import tqdm
import argparse
import json

# Argument Parsing
parser = argparse.ArgumentParser(description="Phrases Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    default='/ssd_scratch/cvit/vanshg/datasets/deaf-youtube/',
    help="Directory of original dataset",
)
parser.add_argument(
    '--speaker',
    type=str,
    default='mia_sandra',
    help='Name of speaker'
)
args = parser.parse_args()

speaker_dir = os.path.join(args.data_dir, args.speaker)
src_clips_dir = os.path.join(speaker_dir, 'website_sentence_clips')
dst_clips_dir = os.path.join(speaker_dir, 'filtered_sentence_clips')

print(f"SPEAKER DIR: {speaker_dir}")
print(f"SRC CLIPS DIR: {src_clips_dir}")
print(f"DST CLIPS DIR: {dst_clips_dir}")

def filter_sentence_clips(src_vid_folder):
    assert os.path.exists(src_vid_folder), f"{src_vid_folder} does not exists"
    video_name = os.path.basename(src_vid_folder)
    dst_vid_folder = os.path.join(dst_clips_dir, f"{video_name}")
    os.makedirs(dst_vid_folder, exist_ok=True)

    # Reading the clips.json for this video
    clip_filepath = os.path.join(src_vid_folder, 'clips.json')
    with open(clip_filepath) as file:
        tracks_clips_data = json.load(file)
    
    # Copy the Video Metadata json files to destination video folder
    shutil.copy(clip_filepath, dst_vid_folder)
    shutil.copy(os.path.join(src_vid_folder, f"tracks.json"), dst_vid_folder)
    shutil.copy(os.path.join(src_vid_folder, f"{video_name}.json"), dst_vid_folder)
    total_video_clips = 0
    accepted_video_clips = 0
    
    # Going through all the face tracks of a video
    for track_clips in tracks_clips_data:
        clips_data = track_clips['clips']

        # Going through all the clips of a track
        for video_clip_data in clips_data:
            total_video_clips += 1
            clip_status = video_clip_data.get('status', 'none')
            
            # Continue if the clip status is not accepted
            if clip_status != 'Accepted':
                continue

            accepted_video_clips += 1
            original_sentence = video_clip_data.get('sentence')
            gt_text = video_clip_data.get('updated_sentence', original_sentence)
            
            video_clip_name = os.path.basename(video_clip_data['clip_output_path']).split('.')[0]
            src_clip_filepath = os.path.join(src_vid_folder, f"{video_clip_name}.mkv")
            assert os.path.exists(src_clip_filepath), f"{src_clip_filepath} does not exists"

            dst_clip_filepath = os.path.join(dst_vid_folder, f"{video_clip_name}.mkv")
            dst_txt_filepath = os.path.join(dst_vid_folder, f"{video_clip_name}.txt")
            # print(f"{dst_clip_filepath = }")
            # print(f"{dst_txt_filepath = }")
   
            # Write the Ground Truth Sentence
            with open(dst_txt_filepath, 'w') as file:
                file.write(f"{dst_clip_filepath} {gt_text}")

            # Copy the video
            shutil.copy(src_clip_filepath, dst_vid_folder)
    
    print(f"\nTotal video clips = {total_video_clips} | Accepted video clips = {accepted_video_clips}\n")

def main(args):
    # video_ids_path = os.path.join(speaker_dir, "all_video_ids.txt")
    # video_ids = open(video_ids_path, 'r').read().split()
    video_ids = os.listdir(src_clips_dir)
    # video_ids = ["vativsC3YgU"]
    video_folders = [os.path.join(src_clips_dir, f"{video_id}") for video_id in video_ids]
    print(f"\nNumber of {len(video_folders) = }")
    print(f"{video_folders[0] = }")
    
    for src_vid_folder in tqdm(video_folders, desc="Filtering Video Clips"):
        print(f"{src_vid_folder = }")
        filter_sentence_clips(src_vid_folder)

if __name__ == '__main__':
    main(args)
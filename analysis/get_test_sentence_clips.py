import os
import json
import glob
import shutil

data_dir = "/ssd_scratch/cvit/vanshg/datasets/accented_speakers"
speaker = "daniel_howell"
speaker_dir = os.path.join(data_dir, speaker)
src_clips_dir = os.path.join(speaker_dir, "sentence_clips")
dst_clips_dir = os.path.join(speaker_dir, "test_sentence_clips")

labels_filepath = os.path.join(speaker_dir, "test_reduced_labels.txt")
test_video_names = []

with open(labels_filepath, 'r') as file:
    for line in file.readlines():
        vid_rel_path = line.split()[0]
        video_name = os.path.basename(vid_rel_path).split('.')[0]
        # print(video_name)
        # print(vid_rel_path)
        test_video_names.append(video_name)

video_names = os.listdir(src_clips_dir)
print(video_names)
# print(test_video_names)

for video_name in video_names:
    src_vid_clips_dir = os.path.join(src_clips_dir, video_name)
    dst_vid_clips_dir = os.path.join(dst_clips_dir, video_name)

    clips_filepath = os.path.join(src_vid_clips_dir, "clips.json")
    with open(clips_filepath, 'r') as file:
        vid_clips = json.load(file)

    os.makedirs(dst_vid_clips_dir, exist_ok=True)

    final_vid_clips = []
    for vid_clip in vid_clips:
        final_vid_clip = {}
        for k, v in vid_clip.items():
            if k != 'clips':
                final_vid_clip[k] = v
                continue
            final_clips = []
            for clip in vid_clip['clips']:
                clip_output_path = clip['clip_output_path']
                clip_name = os.path.basename(clip_output_path).split('.')[0]
                if clip_name in test_video_names:
                    final_clips.append(clip)
                    src_clip_path = os.path.join(src_vid_clips_dir, f"{clip_name}.mp4")
                    src_txt_path = os.path.join(src_vid_clips_dir, f"{clip_name}.txt")
                    shutil.copy(src_clip_path, dst_vid_clips_dir)
                    shutil.copy(src_txt_path, dst_vid_clips_dir)
            
            final_vid_clip['clips'] = final_clips
            # if len(final_clips):
            #     print(final_clips)
        
        final_vid_clips.append(final_vid_clip)
    
    print(final_vid_clips)

    dst_clips_filepath = os.path.join(dst_vid_clips_dir, "clips.json")    
    with open(dst_clips_filepath, 'w') as file:
        json.dump(final_vid_clips, file)

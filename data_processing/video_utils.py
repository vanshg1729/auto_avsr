import os
import math
import subprocess
import cv2
import numpy as np
from tqdm import tqdm

def seconds_to_hhmmss(seconds):
    """
    Convert seconds to hh:mm:ss format.
    
    Args:
        seconds (float): Time in seconds.
    
    Returns:
        str: Time in hh:mm:ss format.
    """
    seconds = math.ceil(seconds * 1e3)/1e3 # Rounding to the nearest decimal of 1e-3
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    # print(f"{seconds = } | {hours = } | {minutes = } | {secs = }")
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def clip_video_ffmpeg(video_path, timestamsp, output_path, copy_stream=False, verbose=False):
    output_dir = os.path.dirname(output_path)

    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    start_time, end_time = timestamsp
    duration = end_time - start_time

    start_time = seconds_to_hhmmss(start_time)
    end_time = seconds_to_hhmmss(end_time)
    duration = seconds_to_hhmmss(duration)
    if verbose:
        print(f"{start_time = } | {end_time = } | {duration = } | {video_path = } | {output_path = }")

    # Construct the ffmpeg command (input of the command)
    command = [
        'ffmpeg',
        '-loglevel', 'panic',           # suppress output except for fatal errors
        '-y',                           # Overwrite output file if it exists
        '-ss', start_time,              # Start time
        '-t', duration,                # Duration
        '-i', video_path,               # Input file
        '-map', '0'                       # Include all the streams
    ]

    # Whether to copy the stream or not
    if not copy_stream:
        command.extend([
            '-c:v', 'libx264',
            '-c:a', 'aac',
        ])
    else:
        command.extend([
            '-c', 'copy',       # Copy video and audio codec
        ])
    
    # Additional options + output_path
    command.extend([
        # '-reset_timestamps', '1',       # Avoid negative timestamps
        # '-avoid_negative_ts', 'make_zero', # Avoid negative timestamps
        output_path
    ])

    if verbose:
        print(command)

    # Run the ffmpeg command
    subprocess.run(command, check=True)

def video_frame_batch_generator(video_path, batch_size):
    """
    Generator function that reads frames from a video and yields them in batches.

    Args:
        video_path (str): Path to the video file.
        batch_size (int): Number of frames per batch.

    Yields:
        tuple: A tuple containing a list of frames and a list of frame indices.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    batch_frames = []
    batch_indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # coverting to RGB
        batch_frames.append(frame)
        batch_indices.append(frame_idx)
        frame_idx += 1

        if len(batch_frames) == batch_size:
            yield batch_indices, batch_frames
            batch_frames = []
            batch_indices = []

    # Yield the last batch if it's not empty
    if batch_frames:
        yield batch_indices, batch_frames

    cap.release()

def save2vid_opencv(filename, vid, fps=25):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    vid = vid.astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    T, H, W, C = vid.shape
    frame_size = (W, H)
    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    for i, frame in enumerate(tqdm(vid, desc="Writing Video frames")):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./images/{i:02d}.png", frame_bgr)
        out.write(frame_bgr)

    out.release()

def split_clip_by_words(clip, max_duration=16):
    """
    Splits a clip with word level segments present according to a max duration threshold 

    Parameters:
    - clip (dict): dictionary containing the keys {'start', 'end', 'sentence', 'words', 'seg_id'}
    - max_duration (int): max duration in seconds for the clip
    """
    start_time = clip['start'] 
    word_segs = clip['words']
    seg_id = clip['seg_id']

    sub_clips = []
    current_start_time = start_time
    current_end_time = start_time
    current_words = []
    current_word_segs = []
    current_duration = 0
    
    for word_seg in word_segs:
        word_start = word_seg.get('start', current_end_time)
        word_end = word_seg.get('end', current_end_time)
        word_text = word_seg['word']
        # print(f"{word_text = } | {word_start = } | {word_end = } | {current_duration = }")
        word_duration = word_end - word_start

        # Start a new sub-clip
        if word_end - current_start_time > max_duration:
            # create a new sub-clip with the current words
            current_sentence = ' '.join(current_words)
            sub_clip = {
                'sentence': current_sentence,
                'start': current_start_time,
                'end': current_end_time,
                'words': current_word_segs,
                'seg_id': seg_id
            }
            sub_clips.append(sub_clip)

            # Start a new clip from this word onwards
            current_start_time = word_start
            current_end_time = word_end
            current_words = [word_text]
            current_word_segs = [word_seg]
            current_duration = current_end_time - current_start_time
        # Continue with the old sub-clip
        else:
            current_end_time = word_end
            current_words.append(word_text)
            current_word_segs.append(word_seg)
            current_duration = current_end_time - current_start_time
    
    # Add the last sub-clip if there is any
    if len(current_words):
        current_sentence = ' '.join(current_words)
        sub_clip = {
            'sentence': current_sentence,
            'start': current_start_time,
            'end': current_end_time,
            'words': current_word_segs,
            'seg_id': seg_id
        }
        sub_clips.append(sub_clip)
    
    return sub_clips

def is_segment_inside_track(face_start, face_end, sent_start, sent_end, tol=0.1):
    """
    Check if a sentence/word segment lies inside a face track segment with a tolerance.

    Parameters:
    - face_start (float): Start time of the face track.
    - face_end (float): End time of the face track.
    - sent_start (float): Start time of the sentence.
    - sent_end (float): End time of the sentence.
    - tol (float): Tolerance in seconds for misalignment (default is 0.1 seconds).

    Returns:
    - bool: True if the sentence lies within the face track considering the tolerance, otherwise False.
    """
    # Check if the sentence start and end times are within the face track times with tolerance
    return (face_start - tol <= sent_start <= face_end + tol) and (face_start - tol <= sent_end <= face_end + tol)

def align_track_to_segments(track, segments, min_clip_len=0.9, word_level=False, max_duration=16.5, verbose=False):
    """
    Gets the Sentence Segments that align with a face track

    Parameters:
    - track (dict): dictionary containing face track metadata {keys: ['start', 'end', 'input_path', 'output_path']}
    - segments (dict): dict containing sentence segments from the input video (WhisperX aligned_segment format)
    - min_clip_length (float): minimum clip length to consider (default = 0.9)
    - word_level (bool): Whether word level timestamps are available so segments can be aligned at word level

    Returns:
    - clips (dict): contains the sentence clips for this face track
    """
    track_st = track['start_time']
    track_end = track['end_time']
    if verbose:
        print(f"Track Start: {track_st} | Track End: {track_end}")

    clips = []
    tol = 0.1 # tolerance in seconds for comparing track and segments
    # clip_template = {"start": , "end": , "sentence": , "words": , "seg_id": }

    # Finding overlapping/contained segments for this face track
    for seg_id, segment in enumerate(segments):
        seg_st = segment['start']
        seg_end = segment['end']

        clip = {}
        # The entire sentence is covered in the face track
        if is_segment_inside_track(track_st, track_end, seg_st, seg_end):

            # Check if the clip satisfies the min length criteria
            if (seg_end - seg_st) + 2 * tol < min_clip_len:
                continue

            clip = {'sentence': segment['text'], 'start': seg_st, 'end': seg_end,
                    'seg_id': seg_id}

            if word_level:
                clip['words'] = segment['words']
            if verbose:
                print(f"ID: {seg_id} | Start: {seg_st} | End: {seg_end} | Sentence: {segment['text']}")
        # The start of the sentence overlaps with the face track
        elif word_level and is_segment_inside_track(track_st, track_end, seg_st, track_end):
            
            if verbose:
                print(f"Start of Segment {seg_id} is overlapping with track")
            all_words = segment['words']
            seg_end = seg_st
            word_segs = []
            words = []
            
            # Finding the list of overlapping words
            for word_id, word in enumerate(all_words):
                # Doing this because WhisperX doesn't give timestamps to numerals like "2014" or "6.1"
                word_st, word_end = word.get('start', seg_end), word.get('end', seg_end)
                if is_segment_inside_track(track_st, track_end, word_st, word_end):
                    if len(words) == 0:
                        seg_st = word_st
                    seg_end = word_end
                    word_segs.append(word)
                    words.append(word['word'])
                else:
                    break
            
            # Check if the clip satisfies the min length criteria
            if (seg_end - seg_st) + 2 * tol < min_clip_len:
                continue
            sentence = ' '.join(words)
            clip = {'sentence': sentence, 'start': seg_st, 'end': seg_end,
                    'words': word_segs, 'seg_id': seg_id}
            if verbose:
                print(f"ID: {seg_id} | Start: {seg_st} | End: {seg_end} | Sentence: {sentence}")
        # The end of the sentence overlaps with the face track
        elif word_level and is_segment_inside_track(track_st, track_end, track_st, seg_end):
            if verbose:
                print(f"End of Segment {seg_id} is overlapping with track")
            all_words = segment['words'][::-1] # starting from the ending words
            seg_st = seg_end
            word_segs = []
            words = []
            
            # Finding the list of overlapping words
            for word_id, word in enumerate(all_words):
                # Doing this because WhisperX doesn't give timestamps to numerals like "2014" or "6.1"
                word_st, word_end = word.get('start', seg_st), word.get('end', seg_st)
                if is_segment_inside_track(track_st, track_end, word_st, word_end):
                    if len(words) == 0:
                        seg_end = word_end
                    seg_st = word_st
                    word_segs.append(word)
                    words.append(word['word'])
                else:
                    break
            
            # Reversing back the list to have the words in ascending order
            words = words[::-1]
            word_segs = word_segs[::-1]
            
            # Check if the clip satisfies the min length criteria
            if (seg_end - seg_st) + 2 * tol < min_clip_len:
                if verbose:
                    print(f"overlap of end of segment {seg_id} is too small | {seg_st = } | {seg_end = }")
                continue
            sentence = ' '.join(words)
            clip = {'sentence': sentence, 'start': seg_st, 'end': seg_end,
                    'words': word_segs, 'seg_id': seg_id}
            if verbose:
                print(f"ID: {seg_id} | Start: {seg_st} | End: {seg_end} | Sentence: {sentence}")
        
        # If a clip exists, then add it to list of clips
        if len(clip.keys()):
            if not word_level:
                clips.append(clip)
                continue

            clip_st = clip['start']
            clip_end = clip['end']
            clip_duration = clip_end - clip_st
            # clip is less than max_duration
            if clip_duration + tol < max_duration:
                clips.append(clip)
            # clip is too long
            else:
                # print(f"\nSplitting clip by words | duration: {clip_duration}")
                sub_clips = split_clip_by_words(clip, max_duration=max_duration)
                # print(f"{len(sub_clips) = }")
                # for sub_clip in sub_clips:
                #     clip_st = sub_clip['start']
                #     clip_end = sub_clip['end']
                #     seg_id = sub_clip['seg_id']
                #     duration = clip_end - clip_st
                #     print(f"{clip_st = } | {clip_end = } | {seg_id = } | {duration = }")
                clips.extend(sub_clips)
                
    return clips

def save_track_clips(face_track, track_id, track_clips, input_vid_dir, output_clip_dir, roundoff=False, verbose=False):
    video_fname = os.path.basename(face_track['input_path']).split('.')[0]
    ext = 'mkv'

    def round_up(number):
        return math.ceil(number * 10) / 10
    
    def round_down(number):
        return math.floor(number * 10) / 10
    
    track_metadata = {
        "track_id": track_id,
        "track_path": face_track['output_path'],
        "track_start": face_track['start_time'],
        "track_end": face_track['end_time'],
        "clips": []
    }

    # Saving all the clips corresponding to a particular face track
    for clip in track_clips:
        clip_st = clip['start']
        clip_end = clip['end']
        # print(f"{clip_st = } | {clip_end = }")
        # print(f"{roundoff = }")
        if roundoff:
            # print(f"INSIDE Roundoff")
            clip_st = round_down(clip_st)
            clip_end = clip_end + 0.1
            clip_end = round_up(clip_end)
        seg_id = clip['seg_id']
        clip_id = clip['clip_id']
        sentence = clip['sentence']

        input_video_path = os.path.join(input_vid_dir, f"{video_fname}.{ext}")
        video_clips_dir = os.path.join(output_clip_dir, f"{video_fname}")
        os.makedirs(video_clips_dir, exist_ok=True)
        output_clip_path = os.path.join(video_clips_dir, f"{video_fname}_{track_id}_{seg_id}_{clip_id}.{ext}")

        # Save the clip 
        clip_video_ffmpeg(input_video_path, (clip_st, clip_end), output_clip_path, verbose=verbose)

        # Save the transcript
        output_txt_path = os.path.join(video_clips_dir, f"{video_fname}_{track_id}_{seg_id}_{clip_id}.txt")
        with open(output_txt_path, 'w') as file:
            file.write(f"{output_clip_path} {sentence}")

        clip_metadata = {
            "start": clip_st,
            "end": clip_end,
            "sentence": sentence,
            "segment_id": seg_id,
            'clip_id': clip_id,
            "clip_output_path": output_clip_path,
        }
        track_metadata['clips'].append(clip_metadata)
    
    return track_metadata
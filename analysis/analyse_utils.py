import os
import re
from collections import Counter
import json
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def process_text(text):
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.upper()
    return text

def get_word_frequencies(sentences):
    # Join all sentences into a single string
    text = ' '.join(sentences)
    
    # Use regex to find all words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Use Counter to count word frequencies
    word_freq = Counter(words)
    
    return word_freq

def get_speaker_df(clip_files):
    videos_metadata = []
    for clip_file_idx, clip_file in enumerate(clip_files):
        # clips of all the tracks
        tracks_clips = json.load(open(clip_file))
        for track_clips in tracks_clips:
            for clip in track_clips['clips']:
                start_time = clip['start']
                end_time = clip['end']
                sentence = clip['sentence']
                processed_text = process_text(sentence)
                num_words = len(sentence.split())
                video_path = clip['clip_output_path']
                video_id = os.path.basename(os.path.dirname(video_path))
                duration = end_time - start_time

                video_data = {
                    'video_path': video_path,
                    'video_id': video_id,
                    'start': start_time,
                    'end': end_time,
                    'transcript': sentence,
                    'processed_text': processed_text,
                    'num_words': num_words,
                    'num_seconds': duration
                }

                videos_metadata.append(video_data)
    
    speaker_df = pd.DataFrame(videos_metadata)
    return speaker_df

def draw_word_cloud(word_freq):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
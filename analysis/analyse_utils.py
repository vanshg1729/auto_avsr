import os
import re
from collections import Counter
import json
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.metrics import edit_distance
import editdistance

def process_text(text):
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.upper()
    return text

def get_word_frequencies(sentences):
    # Join all sentences into a single string
    text = ' '.join(sentences)
    
    # Use regex to find all words
    words = text.lower().split()
    # words = re.findall(r'\b\w+\b', text.lower())
    
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
                clip_status = clip.get('status', 'None')
                start_time = clip['start']
                end_time = clip['end']
                sentence = clip['sentence']
                sentence = clip.get('updated_sentence', sentence)
                processed_text = process_text(sentence)
                num_words = len(sentence.split())
                video_path = clip['clip_output_path']
                video_id = os.path.basename(os.path.dirname(video_path))
                duration = end_time - start_time

                video_data = {
                    'video_path': video_path,
                    'video_id': video_id,
                    'status': clip_status,
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

# Function to calculate substitutions, deletions, and insertions using edit distance DP
def edit_distance_analysis(reference, hypothesis):
    r = reference.split()
    h = hypothesis.split()
    m, n = len(r), len(h)
    
    # Create a DP table
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize the table
    for i in range(1, m+1):
        dp[i][0] = i
    for j in range(1, n+1):
        dp[0][j] = j
    
    # Fill the table
    for i in range(1, m+1):
        for j in range(1, n+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1,  # substitution
                               dp[i-1][j] + 1,    # deletion
                               dp[i][j-1] + 1)    # insertion
    
    # Backtrack to find the operations
    substitutions = []
    deletions = []
    insertions = []
    
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i-1] == h[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions.append((r[i-1], h[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions.append(r[i-1])
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            insertions.append(h[j-1])
            j -= 1
    
    return substitutions, deletions, insertions

# Function to find the most frequent errors across multiple sentence pairs
def aggregate_errors(references, hypotheses):
    all_substitutions = []
    all_deletions = []
    all_insertions = []
    total_errors = 0
    total_words = 0
    
    for reference, hypothesis in zip(references, hypotheses):
        substitutions, deletions, insertions = edit_distance_analysis(reference, hypothesis)
        all_substitutions.extend(substitutions)
        all_deletions.extend(deletions)
        all_insertions.extend(insertions)

        total_errors += len(substitutions) + len(deletions) + len(insertions)
        total_words += len(reference.split())
    
    wer = total_errors/total_words
    return {
        "total_errors": total_errors,
        "total_words": total_words,
        "wer": wer,
        "Substitutions": Counter(all_substitutions),
        "Deletions": Counter(all_deletions),
        "Insertions": Counter(all_insertions)
    }

# Function to print the most frequent errors
def print_most_frequent_errors(result, top_n=5):
    print(f"WER: {result['wer']} | {result['total_errors'] = } | {result['total_words'] = }")
    print("Most Frequent Substitutions:")
    for (ref_word, hyp_word), count in result["Substitutions"].most_common(top_n):
        print(f"'{ref_word}' -> '{hyp_word}': {count} times")
    
    print("\nMost Frequent Deletions:")
    for word, count in result["Deletions"].most_common(top_n):
        print(f"'{word}': {count} times")
    
    print("\nMost Frequent Insertions:")
    for word, count in result["Insertions"].most_common(top_n):
        print(f"'{word}': {count} times")

def main():
    # Example usage with multiple sentences
    references = [
        # "the quick brown fox jumps over the lazy dog",
        # "hello world this is a test",
        "openai is creating safe ai"
    ]

    hypotheses = [
        # "the quick brown cat jumped over a lazy dogs",
        # "hello world this is test",
        "open ai is creating save ai"
    ]

    result = aggregate_errors(references, hypotheses)
    print_most_frequent_errors(result, top_n=3) 

if __name__ == '__main__':
    main()
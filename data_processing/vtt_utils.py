import os
import re
import html

def clean_caption_line(line):
    # Convert HTML entities to their corresponding characters
    line = html.unescape(line)
    
    # Remove text within square brackets (e.g., [Music], [Laughter])
    line = re.sub(r'\[.*?\]', '', line)
    # Remove text within parentheses ()
    line = re.sub(r'\(.*?\)', '', line)
    # Remove text within asterisks **
    line = re.sub(r'\*.*?\*', '', line)
    # Remove text within angle brackets <>
    line = re.sub(r'<.*?>', '', line)
    # Remove single quotes around words or phrases, but not contractions
    line = re.sub(r"(?<!\w)'(\w+(?: \w+)*)'(?!\w)", r"\1", line)
    # A second time because some captions have this quotes 2 times like (''tax'')
    line = re.sub(r"(?<!\w)'(\w+(?: \w+)*)'(?!\w)", r"\1", line)
    # Replace curly single quotes with straight single quotes
    line = re.sub(r'’', "'", line)

    # Remove HTML tags
    line = re.sub(r'<[^>]+>', '', line)
    # Remove timestamps (e.g., 00:00:01.000)
    line = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', line)
    # Remove speaker labels (e.g., SPEAKER: or SPEAKER -)
    line = re.sub(r'^[A-Z]+(\s|-)?:\s?', '', line)
    # Replace HTML entity for 'q&a' with 'qna'
    line = re.sub(r'&', 'n', line)
    # To handle cases like Deaf/HoH or August/September
    line = re.sub(r'/', ' ', line)

    # Note: Maybe also handle cases like Ooo or Erm, Hmm
    # Case : Deaf/HoH (source : XlEO7pWAc84 benny)

    # Remove consecutive non-word characters (e.g., --, ---, …)
    # line = re.sub(r'[^\w\s]+', ' ', line)
    # Clean up any remaining entities like "q&amp;a" to "qna"
    # line = re.sub(r'&[a-z]+;', '', line)
    # Replace triple dots or more with a single dot (e.g., ...)
    line = re.sub(r'\.{3,}', '.', line)
    # Remove non-alphanumeric characters except common punctuation
    line = re.sub(r"[^\w\s,.!?'-]", "", line)
    # Remove extra whitespace
    line = ' '.join(line.split())
    return line

def convert_to_seconds(timestamp):
    h, m, s = timestamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def read_vtt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    entries = []
    current_entry = {}

    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue

        if '-->' in line:
            times = line.split(' --> ')
            current_entry['start'] = times[0]
            current_entry['end'] = times[1]
        elif line:
            if 'text' not in current_entry:
                current_entry['text'] = line
            else:
                current_entry['text'] += ' ' + line
        else:
            if 'start' in current_entry and 'text' in current_entry:
                entries.append(current_entry)
                current_entry = {}

    # Ensure the last entry is added
    if 'start' in current_entry and 'text' in current_entry:
        entries.append(current_entry)
    
    return entries

def process_vtt_entries(entries):
    new_entries = []
    id = 0
    for entry in entries:
        new_entry = {}
        text = clean_caption_line(entry['text'])
        start = convert_to_seconds(entry['start'])
        end = convert_to_seconds(entry['end'])
        if text:
            new_entry.update({'text': text, 'start': start, 'end': end})
            new_entry['id'] = id
            id += 1
            new_entries.append(new_entry)
    
    return new_entries
    
def main():
    import json

    vtt_filepath = '../datasets/deaf-youtube/benny/captions/dyb71EMatR0.en-GB.vtt'
    vtt_file_dir = os.path.dirname(vtt_filepath)
    vtt_filename = os.path.basename(vtt_filepath).split('.')[0]

    json_filepath = os.path.join(vtt_file_dir, f"{vtt_filename}.json")
    entries = read_vtt(vtt_filepath)
    process_vtt_entries(entries)

    with open(json_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(entries, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
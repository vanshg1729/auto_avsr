import json
import random
import os
import numpy as np
import cv2

import torch
import torchaudio
import torchvision
from .transforms import TextTransform


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    vid = torch.tensor(frames)
    vid = vid.permute((0, 3, 1, 2))
    return vid

    # vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    # vid = vid.permute((0, 3, 1, 2))
    # return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class PhraseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        video_transform,
        subset='train',
        data_size=1.0,
        rng_seed=69
    ):

        self.root_dir = root_dir
        self.data_size = data_size
        self.rng_seed = rng_seed
        self.rng = random.Random(rng_seed)
        self.text_transform = TextTransform()

        self.modality = 'video'

        # reading the video list and shuffling it
        self.video_list = self.load_list(label_path)
        # self.rng.shuffle(self.video_list)

        self.subset = subset
        # if subset == 'train':
        #     num_videos = len(self.video_list)
        #     num_train = int(self.data_size * num_videos)
        #     self.video_list = self.video_list[:num_train]
        # elif subset == 'test':
        #     num_videos = len(self.video_list)
        #     num_test = int(self.data_size * num_videos)
        #     self.video_list = self.video_list[-num_test:]
        # else:
        #     raise Exception(f"Invalid subset = {subset}")
        
        print(f"Size of self.video_list: {len(self.video_list)}")

        self.video_transform = video_transform

    def load_list(self, label_path):
        print(f"Label Path : {label_path}, exists = {os.path.exists(label_path)}")
        f = open(label_path, 'r')
        video_list = json.load(f)
        return video_list

    def __getitem__(self, idx):
        video_metadata = self.video_list[idx]
        rel_path = os.path.basename(video_metadata['videoPath'])
        gt_text = video_metadata['transcript'].upper()
        token_id = self.text_transform.tokenize(gt_text)

        path = os.path.join(self.root_dir, "videos", rel_path)
        video = load_video(path)
        video = self.video_transform(video)
        return {"input": video, "target": token_id}

    def __len__(self):
        return len(self.video_list)

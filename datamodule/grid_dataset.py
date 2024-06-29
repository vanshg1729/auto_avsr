import os
import numpy as np
import cv2

import torch
import torchaudio
import torchvision


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


class GridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        video_transform,
        subset='train',
        data_size=1.0,
    ):

        self.root_dir = root_dir
        self.data_size = data_size

        self.modality = 'video'

        self.list = self.load_list(label_path)
        self.subset = subset
        if subset == 'train':
            num_videos = len(self.list)
            num_train = int(self.data_size * num_videos)
            self.list = self.list[:num_train]
        elif subset == 'test':
            num_videos = len(self.list)
            num_test = int(self.data_size * num_videos)
            self.list = self.list[-num_test:]
        else:
            raise Exception(f"Invalid subset = {subset}")
        
        print(f"Size of self.list: {len(self.list)}")

        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            paths_counts_labels.append(
                (
                    dataset_name,
                    rel_path,
                    int(input_length),
                    torch.tensor([int(_) for _ in token_id.split()]),
                )
            )
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        video = load_video(path)
        video = self.video_transform(video)
        return {"input": video, "target": token_id}

    def __len__(self):
        return len(self.list)

#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import torchvision
import cv2
import numpy as np


class AVSRDataLoader:
    def __init__(self, modality, detector="retinaface", convert_gray=True, device="cuda:0"):
        self.modality = modality
        if modality == "video":
            if detector == "retinaface":
                from detectors.retinaface.detector import LandmarksDetector
                from detectors.retinaface.video_process import VideoProcess

                print(f"Device: {device}")
                self.landmarks_detector = LandmarksDetector(device=device)
                self.video_process = VideoProcess(convert_gray=convert_gray)

            if detector == "mediapipe":
                from detectors.mediapipe.detector import LandmarksDetector
                from detectors.mediapipe.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=convert_gray)

    def load_data(self, data_filename, landmarks=None, transform=True):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            return audio
        if self.modality == "video":
            video = self.load_video(data_filename)
            if not landmarks:
                landmarks = self.landmarks_detector(video)
                print(f"Got the face landmarks")
            video = self.video_process(video, landmarks)
            if video is None:
                raise TypeError("video cannot be None")
            video = torch.tensor(video)
            return video

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    # def load_video(self, data_filename):
    #     cap = cv2.VideoCapture(data_filename)
    #     frames = []
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frames.append(frame)
    #     cap.release()
    #     frames = np.array(frames)
    #     return frames

    def load_video(self, data_filename):
        """
        NOTE: There is some issue with torchvision.io.read_video and it doesn't read the entire video
        """
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

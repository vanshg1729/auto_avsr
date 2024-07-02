import os
import json
import glob
import csv

import hydra
import torch
import torchaudio
import torchvision

from datamodule.av_dataset import cut_or_pad
from datamodule.transforms import AudioTransform, VideoTransform

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        if cfg.data.modality in ["audio", "video"]:
            from lightning import ModelModule
        elif cfg.data.modality == "audiovisual":
            from lightning_av import ModelModule
        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
        self.modelmodule.eval()


    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality in ["audio", "audiovisual"]:
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)

        if self.modality in ["video", "audiovisual"]:
            video = self.load_video(data_filename)
            # print(f"load_video shape = {video.shape}")
            landmarks = self.landmarks_detector(video)
            # print(f"Got the video landmarks: {type(landmarks)}, {len(landmarks)}")
            # print(f"type of landmarks[0]: {type(landmarks[0])}, {landmarks[0].shape}")
            video = self.video_process(video, landmarks)
            # print("Pre-processed the video using the landmarks")
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2)) # (T, C, H, W)
            print(f"shape of video = {video.shape}")
            video = self.video_transform(video) # (C, T, H, W)

            print(f"Transformed the input video: {video.shape}")
        if self.modality == "video":
            with torch.no_grad():
                self.modelmodule = self.modelmodule.to(device)
                transcript = self.modelmodule(video)
        elif self.modality == "audio":
            with torch.no_grad():
                transcript = self.modelmodule(audio)

        elif self.modality == "audiovisual":
            print(len(audio), len(video))
            assert 530 < len(audio) // len(video) < 670, "The video frame rate should be between 24 and 30 fps."

            rate_ratio = len(audio) // len(video)
            if rate_ratio == 640:
                pass
            else:
                print(f"The ideal video frame rate is set to 25 fps, but the current frame rate ratio, calculated as {len(video)*16000/len(audio):.1f}, which may affect the performance.")
                audio = cut_or_pad(audio, len(video) * 640)
            with torch.no_grad():
                transcript = self.modelmodule(video, audio)

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

def get_gt_text(video_file_path):
    """
    Gets the Ground truth for GRID
    """
    dir_path = "/ssd_scratch/cvit/vanshg/gridcorpus/transcription/s1"
    fname = os.path.basename(video_file_path).split('.')[0]
    text_fpath = os.path.join(dir_path, f"{fname}.align")
    words = []
    with open(text_fpath, "r") as file:
       for line in file:
           parts = line.split()
           if len(parts) == 3 and parts[2] != 'sil':
               words.append(parts[2])
    return ' '.join(words).upper()

def write_row_to_csv(filename, row):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    # gt_transcript = get_gt_text(cfg.file_path)
    # gt_transcript = "i need my medication"
    gt_transcript = "I WASN'T VERY GOOD AT READING THINGS"
    print(f"Ground Truth Transcript: {gt_transcript}")
    transcript = pipeline(cfg.file_path)
    print(f"transcript: {transcript}")
    wer = compute_word_level_distance(gt_transcript, transcript)/len(gt_transcript.split())
    print(f"WER: {wer}")

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def infer_wildvsr(cfg):
    pipeline = InferencePipeline(cfg)
    f = open('./WildVSR/labels.json', 'r')
    video_dict = json.load(f)
    word_distance = 0
    total_length = 0
    for i, (vid_fname, gt_text) in enumerate(video_dict.items()):
        video_path = f"./WildVSR/videos/{vid_fname}"
        transcript = pipeline(video_path)
        word_distance += compute_word_level_distance(gt_text, transcript)
        total_length += len(gt_text.split())
        print(f"{i} WER: {word_distance/total_length}")
    
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def infer_phrases(cfg):
    pipeline = InferencePipeline(cfg)
    print(f"Got the inference pipeline")
    phrases_dir = "/ssd_scratch/cvit/vanshg/vansh_phrases/preprocessed_phrases/videos"
    # phrases_dir = "/ssd_scratch/cvit/vanshg/vansh_phrases/videos"
    assert os.path.exists(phrases_dir), f"Phrases dir: '{phrases_dir}' doesn't exists"
    print(f"Phrases DIR: {phrases_dir}")
    label_file = "/ssd_scratch/cvit/vanshg/vansh_phrases/test_phrases_30.json"
    print(f"Label file: {label_file}")
    f = open(label_file, 'r')
    video_list = json.load(f)
    print(f"Total number of videos: {len(video_list)}")

    csv_filepath = os.path.join(phrases_dir, "results_test_30.csv")
    csv_fp = open(csv_filepath, "w", newline='')
    writer = csv.writer(csv_fp, delimiter=',')
    row_names = [
        "Filepath",
        "Ground Truth Text",
        "Predicted Text",
        "Length",
        "Word Distance",
        "WER",
        "Total Length",
        "Total Word Distance",
        "Final WER"
    ]
    writer.writerow(row_names)
    print(f"Wrote the first row")
    # write_row_to_csv(csv_filepath, row_names)

    total_word_distance = 0
    total_length = 0
    for i, video_data in enumerate(video_list):
        vid_filepath = video_data['videoPath']
        # dirpath = os.path.dirname(vid_filepath)
        fname = os.path.basename(vid_filepath)
        vid_filepath = os.path.join(phrases_dir, fname)

        # Finding the GT text
        gt_text = video_data['transcript']
        
        # Finding the transcript transcript and WER
        print(f"\n{'*' * 70}")
        transcript = pipeline(vid_filepath)

        wd = compute_word_level_distance(gt_text, transcript)
        gt_len = len(gt_text.split())

        total_word_distance += wd
        total_length += len(gt_text.split())
        wer = total_word_distance/total_length

        data = [
            vid_filepath,
            gt_text,
            transcript,
            gt_len,
            wd,
            wd/gt_len,
            total_length,
            total_word_distance,
            wer
        ]
        writer.writerow(data)

        print(f"{i} GT: {gt_text.upper()}")
        print(f"{i} Pred: {transcript.upper()}")

        print(f"{i} dist = {wd}, len: {gt_len}")
        print(f"{i} WER: {wer}")
        print(f"{'*' * 70}")

    csv_fp.close()

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def infer_lrs3(cfg):
    print(f"Inside the inference LRS3 function")
    pipeline = InferencePipeline(cfg)
    print(f"Got the inference pipeline")
    # lrs3_dir = "/ssd_scratch/cvit/vanshg/test"
    lrs3_dir = "./datasets/lrs3"
    label_file = "./checkpoints/lrs3/labels/test.ref"
    lines = open(label_file).read().splitlines()

    print(f"LRS DIR: {lrs3_dir}")
    # filenames = glob.glob(os.path.join(lrs3_dir, "*/*.mp4"))
    print(f"Total number of videos: {len(lines)}")

    # csv_filepath = os.path.join(lrs3_dir, "results.csv")
    # csv_fp = open(csv_filepath, "w", newline='')
    # writer = csv.writer(csv_fp, delimiter=',')
    # row_names = [
    #     "Filepath",
    #     "Ground Truth Text",
    #     "Predicted Text",
    #     "Length",
    #     "Word Distance",
    #     "WER",
    #     "Total Length",
    #     "Total Word Distance",
    #     "Final WER"
    # ]
    # writer.writerow(row_names)
    # print(f"Wrote the first row")
    # write_row_to_csv(csv_filepath, row_names)

    total_word_distance = 0
    total_length = 0
    for i, line in enumerate(lines):
        basename, gt_text = line.split()[0], " ".join(line.split()[1:])
        data_filename = os.path.join(lrs3_dir, f"{basename}.mp4")

        vid_filepath = data_filename
        # dirpath = os.path.dirname(vid_filepath)
        # fname = os.path.basename(vid_filepath).split('.')[0]
        # txt_filepath = os.path.join(dirpath, f"{fname}.txt")

        # Finding the GT text
        # text_list = open(txt_filepath, 'r').readline().split()[1:]
        # gt_text = ' '.join(text_list)
        
        # Finding the transcript transcript and WER
        print(f"\n{'*' * 70}")
        transcript = pipeline(vid_filepath)

        wd = compute_word_level_distance(gt_text, transcript)
        gt_len = len(gt_text.split())

        total_word_distance += wd
        total_length += len(gt_text.split())
        wer = total_word_distance/total_length

        # data = [
        #     vid_filepath,
        #     gt_text,
        #     transcript,
        #     gt_len,
        #     wd,
        #     wd/gt_len,
        #     total_length,
        #     total_word_distance,
        #     wer
        # ]
        # writer.writerow(data)

        print(f"{i} GT: {gt_text}")
        print(f"{i} Pred: {transcript}")

        print(f"{i} dist = {wd}, len: {gt_len}, cur_wer: {wd/gt_len}")
        print(f"{i} WER: {wer}")
        print(f"{'*' * 70}")

    # csv_fp.close()

if __name__ == "__main__":
    main()

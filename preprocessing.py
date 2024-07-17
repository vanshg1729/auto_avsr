import os
import cv2
import torch
import torchvision
from datamodule.av_dataset import cut_or_pad
from datamodule.transforms import AudioTransform, VideoTransform

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def saveFramesToVideo(frames, savepath, fps=1):
    num_frames, H, W, C = frames.shape
    # You can use other codecs like 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(savepath, fourcc, fps, (W, H))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    cv2.destroyAllWindows()
    print(f"Saving video to {savepath}")
    video.release()


class PreProcessPipeline():
    def __init__(self, detector="mediapipe"):
        self.modality = 'video'
        self.detector = detector

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

    def __call__(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(
            data_filename), f"data_filename: {data_filename} does not exist."

        video = self.load_video(data_filename)
        print(f"Original shape of video : {video.shape}")
        landmarks = self.landmarks_detector(video)
        print(f"Got the video landmarks: {type(landmarks)}, {len(landmarks)}")
        video = self.video_process(video, landmarks)
        print("Pre-processed the video using the landmarks")
        # video = torch.tensor(video)
        # video = video.permute((0, 3, 1, 2))
        print(f"Type of video = {type(video)}")
        print(f"shape of video = {video.shape}")
        return video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

def main():
    filepath = "./mooc/clip1.mp4"
    preprocess_pipeline = PreProcessPipeline()

    video = preprocess_pipeline(filepath)
    saveFramesToVideo(video, './mooc/clip1_processed.mp4', 24)

if __name__ == "__main__":
    main()

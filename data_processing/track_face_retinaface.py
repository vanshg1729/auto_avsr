import os
import sys
import cv2
import numpy as np
import copy
from tqdm import tqdm

from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

device = 'cuda'
model_name = "resnet50"
face_detector = RetinaFacePredictor(device=device, threshold=0.8, model=RetinaFacePredictor.get_model(model_name))

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)

def euclidean_distance(vector1, vector2):
    dist = vector2 - vector1
    dist = np.sum(dist * dist)
    dist = np.sqrt(dist)
    return dist

def draw_face_tracks(frames, face_detector):
    frames_with_track = []    
    for frame_id, frame in enumerate(tqdm(frames)):
        frame_copy = copy.deepcopy(frame)
        detected_faces = face_detector(frame)
        if len(detected_faces):
            detected_face = detected_faces[0]
            (x1, y1, x2, y2) = detected_face[:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        # if len(bboxes[0]):
        #     for box_id, bbox in enumerate(bboxes[0]):
        #         # bbox = bboxes[0][0]
        #         (x1, y1, x2, y2) = bbox
        #         face = frame_copy[y1:y2, x1:x2]
        #         cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        frames_with_track.append(frame_copy)
    
    return frames_with_track

def save2vid_opencv(filename, vid, fps=25):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    vid = vid.astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    T, H, W, C = vid.shape
    frame_size = (W, H)
    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    for i, frame in enumerate(vid):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./images/{i:02d}.png", frame_bgr)
        out.write(frame_bgr)

    out.release()

video_path = "/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/dl/videos/a4k8YorzUdI.mp4"
# video_path = "../datasets/Lip2Wav/chem/face_tracks/OSts9bfX6cA/track-2.mp4"
# video_path = './deafs.mp4'

frames = []
cap = cv2.VideoCapture(video_path)
while len(frames) < 900:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

frames = np.array(frames)
print(f"{len(frames) = }")
frames_with_tracks = draw_face_tracks(frames, face_detector)
frames_with_tracks = np.array(frames_with_tracks)
# print(frames_with_tracks.shape)
save2vid_opencv("test.mp4", frames_with_tracks)
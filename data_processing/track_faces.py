import os
import sys
import cv2
import numpy as np
import copy
from tqdm import tqdm

from deepface import DeepFace

# Yolo Face Detector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preparation.detectors.yoloface.face_detector import YoloDetector
face_detector = YoloDetector(device=f"cuda:0", min_face=25)

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
        bboxes, points = face_detector.predict(frame)
        frame_copy = copy.deepcopy(frame)
        if len(bboxes[0]):
            for box_id, bbox in enumerate(bboxes[0]):
                # bbox = bboxes[0][0]
                (x1, y1, x2, y2) = bbox
                face = frame_copy[y1:y2, x1:x2]
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                face_emb = DeepFace.represent(img_path=face_bgr, enforce_detection=False)
                face_emb = face_emb[0]['embedding']
                similarity = cosine_similarity(embedding, face_emb)
                if frame_id == 16:
                    print(f"{similarity = }")
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                    cv2.imwrite(f"{box_id}.png", face_bgr)
                if similarity < 0.5:
                    continue
                print(f"general {similarity = }")
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

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

# video_path = "../datasets/Lip2Wav/chem/face_tracks/OSts9bfX6cA/track-2.mp4"
video_path = './deafs.mp4'
embedding = DeepFace.represent(img_path="./benny.png")[0]['embedding']
embedding = np.array(embedding)
print(f"Shape of embedding: {embedding.shape}")

frames = []
cap = cv2.VideoCapture(video_path)
while len(frames) < 20:
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
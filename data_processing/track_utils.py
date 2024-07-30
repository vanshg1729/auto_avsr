import os
import sys

import cv2
import numpy as np
from tqdm import tqdm
from video_utils import save2vid_opencv

def get_batch_prediction_retinaface(frames, face_detector):
    preds = []
    for frame_idx, frame in enumerate(frames):
         pred = face_detector(frame)
         preds.append(pred)
    
    return preds

def get_bboxes_from_retina_preds(preds):
    """
    Parameters:
    - preds (list): list of numpy array containing (N_i, 15) retinaface prediction

    Returns:
    - bboxes (list): list of numpy array containing (N_i, 4) bounding boxes
    """
    bboxes = []
    for pred_id, pred in enumerate(preds):
        frame_bboxes = []
        for bbox in pred:
            (x1, y1, x2, y2) = bbox[:4]
            w, h = (x2 - x1), (y2 - y1)
            new_bbox = (x1, y1, w, h)
            frame_bboxes.append(new_bbox)

        frame_bboxes = np.array(frame_bboxes)
        bboxes.append(frame_bboxes)
    
    return bboxes

def get_batch_prediction_yolov5(frames, face_detector):
     frames_bboxes = []
     bboxes, points = face_detector.predict(frames)

     for frame_bboxes in bboxes:
        frame_boxes = []
        for bbox in frame_bboxes:
            if not len(bbox):
                continue
            (x1, y1, x2, y2) = bbox
            w, h = (x2 - x1), (y2 - y1)
            new_bbox = (x1, y1, w, h)
            frame_boxes.append(new_bbox)
        frame_boxes = np.array(frame_boxes)
        frames_bboxes.append(frame_boxes)

     return frames_bboxes

def draw_bboxes(frames, bboxes):
    frames_with_bbox = np.copy(frames)
    for frame_idx, frame in enumerate(tqdm(frames_with_bbox, desc="Drawing Bounding Boxes")):
        frame_bboxes = bboxes[frame_idx]
        for bbox in frame_bboxes:
            (x1, y1, w, h) = bbox
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            x1, y1 = int(x1), int(y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    
    return frames_with_bbox

def main():
    video_path = '../datasets/deaf-youtube/benny/videos/JaB9BT09nSE.mp4'
    
    # Reading the frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < 2000:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        # if len(frames) == 900:
        #     frames = frames[1:]
    cap.release()
    frames = np.array(frames)
    print(f"{len(frames) = }")

    # RetinaFace detector
    from ibug.face_alignment import FANPredictor
    from ibug.face_detection import RetinaFacePredictor

    device = 'cuda'
    model_name = "resnet50"
    face_detector = RetinaFacePredictor(device=device, threshold=0.8, 
                                        model=RetinaFacePredictor.get_model(model_name))
    
    print(f"Got RetinaFace detector")

    preds = get_batch_prediction_retinaface(frames, face_detector)
    print(f"Got RetinaFace Preds")
    bboxes = get_bboxes_from_retina_preds(preds)

    frames_with_bbox = draw_bboxes(frames, bboxes)
    save2vid_opencv('benny.mp4', frames_with_bbox)

if __name__ == '__main__':
    main()
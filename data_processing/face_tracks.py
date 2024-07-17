import numpy as np
from tqdm import tqdm
import cv2

from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

def get_single_face_tracks(frames, device):
    """
    This tracker assumes that there will only be a single person/face in the entire video
    This uses the RetinaFace detector
    """
    tracks = []
    
    model_name = "resnet50"
    face_detector = RetinaFacePredictor(device=device, threshold=0.8,
                                        model=RetinaFacePredictor.get_model(model_name))

    total_frames = len(frames)

    for idx in tqdm(range(total_frames), desc="Processing Frames"):
        frame = frames[idx]
        detected_faces = face_detector(frame)

        # Continue if no face is detected in the frame
        num_detections = len(detected_faces)
        if num_detections == 0:
            continue

        # Detected face along with bounding box
        detected_face = detected_faces[0]
        (x1, y1, x2, y2) = detected_face[:4]
        w, h = (x2 - x1), (y2 - y1)
        bbox = (x1, y1, w, h)

        # Check the already existing tracks
        create_new_track = True
        if len(tracks):
            last_track = tracks[-1]
            last_track_frame = last_track['end_frame']

            # Continue the previous track
            if idx == last_track_frame + 1:
                last_track['end_frame'] = idx
                last_track['frames'].append(frame)
                last_track['bboxes'].append(bbox)
                create_new_track = False
        
        # Start a new track
        if create_new_track:
            new_track = {
                "start_frame": idx,
                "end_frame": idx,
                "frames": [frame],
                "bboxes": [bbox]
            }
            tracks.append(new_track)
    
    del face_detector

    return tracks

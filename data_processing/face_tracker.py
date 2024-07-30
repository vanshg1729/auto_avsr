import os
import numpy as np
import copy
import cv2

class FaceTracker:
    def __init__(self, iou_threshold=0.3, min_frames_thresh=25, verbose=False):
        self.iou_threshold = iou_threshold
        self.min_frames_thresh = min_frames_thresh
        self.verbose = verbose
        self.active_tracks = []
        self.finished_tracks = []
        self.saved_tracks = []
        self.next_id = 0
        self.next_save_id = 0
    
    def update(self, detections, frame_idx):
        """
        Parameters:
            - detections (numpy.ndarray): array of shape (num_detections, 4)
            - frame_idx
        """
        # Step 0: All detections become active tracks if there were no active tracks
        if len(self.active_tracks) == 0:
            for detection in detections:
                new_track = self._start_new_track(detection, frame_idx)
                self.active_tracks.append(new_track)
            return

        # Step 1: Update active tracks with new detections
        unmatched_detections = list(detections)
        matched_tracks = []
        unmatched_tracks = []
        new_tracks = []

        # Match each active track to the best new detection using IoU
        for track in self.active_tracks:
            best_match = None
            best_iou = 0
            track_last_frame = track['end_frame']
            for detection_idx, detection in enumerate(detections):
                iou = self._iou(track['bbox'], detection)
                # if self.verbose:
                #     print(f"{frame_idx = } | {track['id'] = } {detection_idx = } | {iou = }")
                if iou > best_iou and iou > self.iou_threshold:
                    best_match = detection
                    best_iou = iou
            
            # This track got a detection matched
            if best_match is not None and frame_idx == track_last_frame + 1:
                # update the track and remove detection from list
                track['end_frame'] = frame_idx
                track['history'].append(best_match)
                track['bbox'] = best_match
                matched_tracks.append(track)

                pop_idx = None
                for detection_idx, detection in enumerate(unmatched_detections):
                    if np.allclose(detection, best_match):
                        pop_idx = detection_idx

                if pop_idx is not None:
                    unmatched_detections.pop(pop_idx)
            else:
                unmatched_tracks.append(track)
        
        # Step 2: Handle all the unmatched tracks
        speaker_track = self._find_speaker_track()
        speaker_track_id = -1 if speaker_track is None else speaker_track['id']
        for track in unmatched_tracks:
            # This face track is of the active speaker
            if track['id'] == speaker_track_id:
                track['saved'] = False
                track['save_id'] = self.next_save_id
                self.saved_tracks.append(track)
                self.finished_tracks.append(track)
                self.next_save_id += 1
                if self.verbose:
                    print(f"Saved track with {track['id'] = } | save_id = {track['save_id']} | {speaker_track_id = }")
            # This face track does not belong to the active speaker
            else:
                self.finished_tracks.append(track)
                if self.verbose:
                    start_frame = track['start_frame']
                    end_frame = track['end_frame']
                    num_frames = end_frame - start_frame + 1
                    print(f"Finished track with {track['id'] = } | {speaker_track_id = } | {start_frame = } | {end_frame = } | {num_frames = }")

        # Step 3 : Create a new track for each of the unmatched detections
        for detection in unmatched_detections:
            track = self._start_new_track(detection, frame_idx)
            new_tracks.append(track)

        # Create the new set of active tracks
        self.active_tracks = copy.deepcopy(matched_tracks + new_tracks)
    
    def _start_new_track(self, detection, frame_idx):
        new_track = {'id': self.next_id, 'bbox': detection, 
                     'start_frame': frame_idx, 'end_frame': frame_idx,
                     'history': [detection]}
        if self.verbose:
            print(f"Started a new Track at {frame_idx = } with id = {self.next_id} | {len(self.active_tracks) = }")

        self.next_id += 1
        return new_track
    
    def _iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter_area = 0

        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0
    
    def _find_speaker_track(self):
        max_frames_track = None
        max_frames_in_track = 0
        for track in self.active_tracks:
            start_frame = track['start_frame']
            end_frame = track['end_frame']
            num_frames = end_frame - start_frame + 1

            if max_frames_in_track < num_frames and self.min_frames_thresh < num_frames:
                max_frames_track = track
                max_frames_in_track = num_frames

        return max_frames_track

def draw_track_on_frames(frames, face_track):
    frames_with_track = np.copy(frames)
    bboxes = face_track['history']
    start_frame = face_track['start_frame']
    track_id = face_track['id']

    for i, bbox in enumerate(bboxes):
        frame_idx = start_frame + i
        frame = frames_with_track[frame_idx]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # Put the track ID
        cv2.putText(frame, f'{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frames_with_track

def main():
    import sys
    from video_utils import save2vid_opencv
    from track_utils import get_batch_prediction_retinaface, get_bboxes_from_retina_preds
    from track_utils import get_batch_prediction_yolov5

    # video_path = '../datasets/deaf-youtube/benny/videos/JaB9BT09nSE.mp4'
    # video_path = '../datasets/Lip2Wav/chem/videos/s_xlDaR53v0.mp4'
    video_path = './deafs.mp4'

    detector = 'yolov5'
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    # Get the total number of frames and fps
    num_frames = 10
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Reading the frames
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        # if len(frames) == 900:
        #     frames = frames[1:]
    cap.release()
    frames = np.array(frames)

    num_frames = len(frames)
    print(f"{len(frames) = }")

    # RetinaFace detector
    if detector == 'retinaface':
        from ibug.face_alignment import FANPredictor
        from ibug.face_detection import RetinaFacePredictor
        device = 'cuda'
        model_name = "resnet50"
        face_detector = RetinaFacePredictor(device=device, threshold=0.8, 
                                            model=RetinaFacePredictor.get_model(model_name))
    elif detector == 'yolov5':
        sys.path.append(os.path.abspath('..'))
        from preparation.detectors.yoloface.face_detector import YoloDetector
        face_detector = YoloDetector(device='cuda:0', min_face=25)
    else:
        print(f"Detector {detector} not found")
        exit(0)
    
    print(f"Got the face detector")
    
    if detector == 'retinaface':
        # Getting the frame predictions
        preds = get_batch_prediction_retinaface(frames, face_detector)
        print(f"Got RetinaFace Preds")
        frames_bboxes = get_bboxes_from_retina_preds(preds) # List of Numpy array
    elif detector == 'yolov5':
        frames_temp = list(frames)
        frames_bboxes = get_batch_prediction_yolov5(frames_temp, face_detector)

    # Face Tracking
    face_tracker = FaceTracker(iou_threshold=0.5, verbose=True)
    for frame_idx, frame in enumerate(frames):
        # Face bounding boxes for this frame 
        detections = frames_bboxes[frame_idx]

        # Update the face tracker
        face_tracker.update(detections, frame_idx)

        # Update the tracker again in case of last frame
        if frame_idx == num_frames - 1:
            face_tracker.update([], frame_idx + 1)
    
    print(f"Number of tracks = {len(face_tracker.finished_tracks)}")
    frames_with_tracks = np.copy(frames)
    for i, track in enumerate(face_tracker.finished_tracks):
        frames_with_tracks = draw_track_on_frames(frames_with_tracks, track)

    # frames_with_bbox = draw_bboxes(frames, bboxes)
    save2vid_opencv('test.mp4', frames_with_tracks)

if __name__ == '__main__':
    main()
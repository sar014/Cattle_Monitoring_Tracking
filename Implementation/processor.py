import cv2
import joblib
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import calculate_features

class CattleAnalyzer:
    def __init__(self, detect_model, pose_model, rf_model, scaler):
        self.yolo = YOLO(detect_model)
        self.pose = YOLO(pose_model)
        self.rf = joblib.load(rf_model)
        self.scaler = joblib.load(scaler)
        self.tracker = DeepSort(max_age=30)
        self.history = {} # Store {track_id: center_point}
        self.label_map = {0: 'Walking', 1: 'Eating', 2: 'Standing', 3: 'Distress', 4: 'Lying'}

    def process_frame(self, frame):
        results = self.yolo(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            if conf > 0.5:
                detections.append(([float(x1), float(y1), float(x2-x1), float(y2-y1)], conf, 'cow'))

        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_behaviors = []

        for track in tracks:
            if not track.is_confirmed(): continue
            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            
            # Pose Estimation
            crop = frame[max(0,t):b, max(0,l):r]
            if crop.size == 0: continue
            
            p_res = self.pose(crop, verbose=False)[0]
            if p_res.keypoints is None or len(p_res.keypoints.xy) == 0: continue
            
            kpts = p_res.keypoints.xy[0].cpu().numpy()
            curr_center = np.array([(l+r)/2, (t+b)/2])
            prev_center = self.history.get(tid)
            
            # Feature Extraction & Prediction
            raw_feats = calculate_features(kpts, prev_center, curr_center)
            scaled_feats = self.scaler.transform([raw_feats])
            label = self.rf.predict(scaled_feats)[0]
            behavior_name = self.label_map[label]
            
            self.history[tid] = curr_center
            active_behaviors.append({"id": tid, "behavior": behavior_name})

            # Draw on frame
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}: {behavior_name}", (l, t-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, active_behaviors
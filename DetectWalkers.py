import cv2
import numpy as np
import torch
import torchreid
import torchvision.transforms as T
from collections import deque
from ultralytics import YOLO
from TrackFaces import CharacterTracker
from InOrOutCar import InCarVisualVerifier

pose_model = YOLO("YoloModels/yolo11n-pose.pt")
vehicle_model = YOLO("YoloModels/yolo11n.pt")
START_TIME_OFFSET = 446 #314 or 3028 or 329 or 446
START_PADDING = 2.0

def seconds_to_cue(seconds, fps=30.0) :
    total_frames = int(round(seconds * fps))
    m = total_frames // (int(fps) * 60)
    s = (total_frames // int(fps)) % 60
    f = int(((total_frames % int(fps)) / fps) * 75)
    return f"{m:02d}:{s:02d}:{f:02d}"

def write_cue(file_obj, track_num, char_id, start_sec, end_sec, fps_val) : 
    start_tc = seconds_to_cue(start_sec, fps_val)
    end_tc = seconds_to_cue(end_sec, fps_val)
    
    file_obj.write(f'  TRACK {track_num:02d} AUDIO\n')
    file_obj.write(f'    TITLE "{char_id} Walk Action"\n')
    file_obj.write(f'    PERFORMER "{char_id}"\n')
    file_obj.write(f'    INDEX 01 {start_tc}\n')
    file_obj.write(f'    REM END {end_tc}\n')
    file_obj.write(f'    REM DURATION {end_sec - start_sec:.2f}s\n')
    file_obj.flush()

class ShotDetector : 
    def __init__(self, threshold=30.0, b_thresh=0.65, cooldown_frames=15) :
        self.prev_hist, self.prev_gray = None, None
        self.threshold = threshold
        self.cooldown_frames = cooldown_frames
        self.frames_since_cut = cooldown_frames
        self.b_thresh = b_thresh
        
    def detect_cut(self, frame) : 
        self.frames_since_cut += 1
        
        small_frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        is_cut = False
        if self.prev_hist is not None and self.prev_gray is not None:
            if self.frames_since_cut >= self.cooldown_frames :
                b_dist = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                l_diff = np.mean(cv2.absdiff(self.prev_gray, gray))
                if b_dist > self.b_thresh and l_diff > self.threshold : 
                    is_cut = True
                    self.frames_since_cut = 0
            
        self.prev_hist = hist
        self.prev_gray = gray
        return is_cut

class KalmanBoxTracker : 
    def __init__(self, bbox) : 
        self.kf = cv2.KalmanFilter(8, 4)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        cx, cy, w, h = self.bbox_to_z(bbox)
        self.kf.statePost = np.array([[cx], [cy], [w], [h] , [0], [0], [0], [0]],  dtype=np.float32)
        self.time_since_update =0
    
    @staticmethod
    def bbox_to_z(bbox) : 
        x1, y1, x2, y2 = bbox
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        return np.array([x1 + w/2, y1 + h/2, w, h], dtype=np.float32)
    
    @staticmethod
    def z_to_bbox(z) :
        cx, cy, w, h = z[:4]
        return [float(cx - w / 2.0), float(cy - h / 2.0), float(cx + w / 2.0), float(cy + h / 2.0)]
    
    def predict(self, camera_shift=(0.0, 0.0)) :
        self.kf.predict()
        
        dx, dy = camera_shift
        self.kf.statePre[0] -= dx
        self.kf.statePre[1] -= dy
        
        self.kf.statePost = self.kf.statePre.copy()
        self.time_since_update += 1
        return self.z_to_bbox(self.kf.statePost[:4].flatten())
    
    def update(self, bbox) : 
        z = self.bbox_to_z(bbox)
        self.kf.correct(z.reshape(4, 1))
        self.time_since_update = 0
        return self.z_to_bbox(self.kf.statePost[:4].flatten())
        
    def get_box(self) : return self.z_to_bbox(self.kf.statePost[:4].flatten())

def computer_iou(box1, box2) : 
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(b1_area + b2_area - inter_area + 1e-6)

def extract_keypoints(kps, frame_shape=None, margin=10) :
    if len(kps) < 17 : return None
    
    extracted = {}
    key_names = {11: "l_hip", 12: "r_hip", 13: "l_knee", 14:"r_knee", 15: "l_ankle", 16: "r_ankle", 5: "l_shoulder", 6: "r_shoulder"}
    h_frame, w_frame = frame_shape[:2] if frame_shape is not None else (None, None)
    
    for idx, name in key_names.items() : 
        threshold = 0.20 if "shoulder" in name else 0.30
        if len(kps[idx]) < 3 or kps[idx][2] <= threshold :  continue
        x, y = float(kps[idx][0]), float(kps[idx][1])
        if frame_shape is not None and (x < margin or x >= w_frame - margin or y < margin or y >= h_frame - margin): continue 
        extracted[name] = kps[idx][:2]
    if not extracted : return None, False
    valid = ({"l_shoulder", "r_shoulder"} <= extracted.keys() or {"l_hip", "r_hip"} <= extracted.keys() or {"l_knee", "r_knee"} <= extracted.keys() or {"l_ankle", "r_ankle"} <= extracted.keys())
    if not valid : return None, False
    
    required_gait = {"l_hip", "r_hip", "l_ankle", "r_ankle"}
    is_valid_gait = required_gait <= extracted.keys()        
    return extracted, is_valid_gait

class AppearenceReID : 
    def __init__(self, model_name="osnet_x1_0") : 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = torchreid.models.build_model(name=model_name, num_classes=1000, loss="softmax", pretrained=True)
        self.feature_extractor.to(self.device).eval()
        self.transform = T.Compose([ T.ToPILImage(), T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    @torch.no_grad()
    def extraction(self, crop) : 
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10 : return None
        
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        features = self.feature_extractor(input_tensor).squeeze(0)
        
        norm = torch.norm(features, p = 2)
        if norm > 0 : features = features / norm
        return features.cpu().numpy()
    
    @staticmethod
    def compare_embeddings(emb1, emb2):
        if emb1 is None or emb2 is None: return 0.0
        return max(0.0, float(np.dot(emb1, emb2)))
        
class CrossCutTracker:
    def __init__(self, match_thresh=0.65) : 
        self.reid = AppearenceReID()
        self.match_thresh = match_thresh
        self.active_gallery = {}
        
        self.w_appearance = 0.85
        self.w_position = 0.10
        self.w_velocity = 0.05
        
    def update(self, track_id, crop, box, velocity=(0.0, 0.0)) :
        emb = self.reid.extraction(crop)
        if emb is not None : 
            centroid = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            if track_id in self.active_gallery :
                old_emb = self.active_gallery[track_id]["embedding"]
                h_crop, w_crop = crop.shape[:2]
                if h_crop > 1.5 * w_crop :
                    updated_emb = 0.90 * old_emb + 0.10 * emb
                    updated_emb = updated_emb / (np.linalg.norm(updated_emb) + 1e-6)
                    self.active_gallery[track_id] = {"embedding" : updated_emb, "last_pos" : centroid, "velocity" : velocity}
            else : self.active_gallery[track_id] = {"embedding" : emb, "last_pos" : centroid, "velocity" : velocity}
                
    def find_match(self, new_crop, new_box, camera_shift, frame_shape, exclude_ids, kalman_tracker, is_shot_cut=False) : 
        new_emb = self.reid.extraction(new_crop)
        if new_emb is None : return None
        
        h_frame, w_frame = frame_shape[:2]
        new_centroid = np.array([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
        scores = []
        
        for track_id, data in self.active_gallery.items():
            sim = self.reid.compare_embeddings(new_emb, data["embedding"])
            if sim < 0.45 or track_id in exclude_ids: continue
            
            if is_shot_cut : score = sim
            else :
                if kalman_tracker  and track_id in kalman_tracker :
                    k_box = kalman_tracker[track_id].get_box()
                    predicted_pos = np.mean([[k_box[0], k_box[1]], [k_box[2], k_box[3]]], axis=0)
                else : predicted_pos = data['last_pos'] - camera_shift
            
                dist = np.linalg.norm(new_centroid - predicted_pos)
                max_diag = np.sqrt(w_frame**2 + h_frame**2)
                s_pos = max(0.0, 1.0 - (dist / (max_diag * 0.2)))
                score = (self.w_appearance * sim + (1.0 - self.w_appearance) * s_pos)
            
            scores.append((score, track_id))
         
        if not scores : return None        
        best_score, best_id = max(scores)
        if best_score < self.match_thresh : return None
        
        if len(scores) > 1 : 
            second_best = scores[1][0]
            if (best_score - second_best) < 0.10 : return None
                  
        return best_id
    
class BackgroundMotion:
    def __init__(self) : self.prev_gray = None
   
    def estimate_shift(self, frame, person_boxes) :
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        
        if self.prev_gray is None : 
            self.prev_gray = current_gray
            return 0.0, 0.0

        if self.prev_gray.shape != current_gray.shape : self.prev_gray = cv2.resize(self.prev_gray, (current_gray.shape[1], current_gray.shape[0]))
        
        bg_mask = np.full(current_gray.shape, 255, dtype=np.uint8)
        h, w = current_gray.shape[:2]
        
        for box in person_boxes : 
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1 : bg_mask[y1:y2, x1:x2] = 0
            
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, current_gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        self.prev_gray = current_gray
        flow_vectors = flow[bg_mask > 0]
        if len(flow_vectors) == 0 : return 0.0, 0.0
        return float(np.median(flow_vectors[:, 0])), float(np.median(flow_vectors[:, 1]))
        
class MovementTracker:
    def __init__(self, buffer_size=40, tau_enter=0.25, tau_exit=0.05):
        self.buffer_size = buffer_size
        self.tau_enter = tau_enter
        self.tau_exit = tau_exit
        
        self.score_history = deque(maxlen=10)
        for dequeing in ["pos_history", "left_foot", "right_foot", "gait_history", "scale_history", "height_history", "shoulder_history"] : setattr(self, dequeing, deque(maxlen=buffer_size))
        self.smoothed_kps = {}
        self.current_state = "STANDING"
        
        self.smoothed_scale = None
        self.last_centroid = None
        self.last_frame_seen = -1
        self.current_position = None
    
    def is_near_edge(self, box, frame_shape, margin=15):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        return x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >=(h - margin)
    
    def get_center_and_scale(self, keypoints, fallback_box) :
        cx, cy, scale = None, None, None
        
        if keypoints : 
            if keypoints and "l_hip" in keypoints and "r_hip" in keypoints : 
                hip_mid = (np.array(keypoints["l_hip"]) + np.array(keypoints["r_hip"])) / 2.0
                cx, cy = hip_mid[0], hip_mid[1]
            elif keypoints and "l_shoulder" in keypoints and "r_shoulder" in keypoints :
                sh_mid = (np.array(keypoints["l_shoulder"]) + np.array(keypoints["r_shoulder"])) / 2.0
                cx, cy = sh_mid[0], sh_mid[1]
                
            if "l_shoulder" in keypoints and "r_shoulder" in keypoints and "l_hip" in keypoints and "r_hip" in keypoints :
                sh_mid = (np.array(keypoints["l_shoulder"]) + np.array(keypoints["r_shoulder"])) / 2.0
                hip_mid = (np.array(keypoints["l_hip"]) + np.array(keypoints["r_hip"])) / 2.0
                scale = np.linalg.norm(sh_mid - hip_mid)
        
        if cx is None or cy is None:
            cx = (fallback_box[0] + fallback_box[2]) / 2.0
            cy = (fallback_box[1] + fallback_box[3]) / 2.0
        
        if scale is None or scale < 5.0 : scale = float(fallback_box[3] - fallback_box[1]) * 0.4
        
        if self.smoothed_scale is None : self.smoothed_scale = scale
        else : self.smoothed_scale = 0.85 * self.smoothed_scale + 0.15 * scale
        
        return (cx, cy), max(10.0, self.smoothed_scale)

    def reset(self):
        for clearings in (self.pos_history, self.left_foot, self.right_foot, self.gait_history, self.height_history, self.shoulder_history, self.scale_history, self.smoothed_kps, self.score_history) : clearings.clear()
        self.current_state = "STANDING"
        self.last_centroid, self.smoothed_scale, self.current_position = None, None, None
        self.last_frame_seen = -1
    
    def evaluate_gait(self, disp_score=0.0) : 
        if disp_score < 0.08 or len(self.gait_history) < 15 : return 0.0
        signal = np.asarray(self.gait_history, dtype=np.float32)
        signal = signal - np.median(signal)
        
        p2p_amp = np.percentile(signal, 90) - np.percentile(signal, 10)
        if p2p_amp < 0.040 : return 0.0
        
        if np.std(signal) < 1e-6 : return 0.0
        
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr /= autocorr[0]
        
        peak = np.max(autocorr[2:len(autocorr)//2])
        periodicity = np.clip((peak - 0.2) / 0.6, 0.0, 1.0)
        amplitude = np.clip((p2p_amp - 0.04) / 0.12, 0.0, 1.0)
        return float(0.5 * amplitude + 0.5 * periodicity)
    
    def update(self, box, frame_shape, camera_shift, frame_idx, keypoints=None, is_occluded=False, char_id = None, is_shot_cut=False, is_valid_gait=False, person_position="Outside") :
        h_frame, w_frame = frame_shape[:2]
        if (h_frame == 0 or w_frame == 0) and self.current_state != "WALKING": return "ANALYZING", 0.0, None, 0.0, 0.0
                
        cam_dx, cam_dy = camera_shift
        (cx, cy), body_scale = self.get_center_and_scale(keypoints, box)
        
        if is_shot_cut or (self.last_centroid is not None and np.linalg.norm(np.array([cx, cy]) - np.array(self.last_centroid)) > 2.5 * body_scale): self.reset()
        if len(self.pos_history) == 0 and self.is_near_edge(box, frame_shape) : return self.current_state, 0.0, None, 0.0, 0.0
        
        bbox_height = float(box[3] - box[1])
        self.height_history.append(bbox_height / (body_scale + 1e-6))        
        
        if is_occluded : 
            self.last_frame_seen = frame_idx
            return self.current_state, 0.0, None, 0.0, 0.0
        
        if self.last_frame_seen is not None and (frame_idx - self.last_frame_seen) > 25 : self.reset()
        
        if self.last_centroid is not None and self.current_position is not None:
            step_dx = (cx - self.last_centroid[0]) - cam_dx
            step_dy = (cy - self.last_centroid[1]) - cam_dy
            self.current_position = (self.current_position[0] + step_dx, self.current_position[1] + step_dy)
        else : self.current_position = (cx, cy)
        
        self.pos_history.append(self.current_position)
        self.scale_history.append(body_scale)
        self.last_centroid = (cx, cy)
        self.last_frame_seen = frame_idx
        
        if keypoints:
            smoothed = {}
            for name, pt in keypoints.items() :
                if name in self.smoothed_kps : smoothed[name] = 0.35 * np.array(pt) +  0.65 * self.smoothed_kps[name]
                else : smoothed[name] = np.array(pt)
            
            self.smoothed_kps = smoothed
            kps = self.smoothed_kps
            
            if is_valid_gait and "l_hip" in kps and "r_hip" in kps and "l_ankle" in kps and "r_ankle" in kps : 
                hip_mid = (np.array(kps['l_hip']) + np.array(kps["r_hip"])) / 2.0
                l_rel, r_rel = (np.array(kps["l_ankle"]) - hip_mid) / (body_scale + 1e-6), (np.array(kps["r_ankle"]) - hip_mid) / (body_scale + 1e-6)
                self.left_foot.append(l_rel)
                self.right_foot.append(r_rel)
                self.gait_history.append(float(l_rel[1] - r_rel[1]))
               
        if len(self.pos_history) < 10 : 
            return self.current_state, self.score_history[-1] if self.score_history else 0.0, None, 0.0, 0.0
        
        positions = np.asarray(self.pos_history, dtype=np.float32)
        if len(positions) >= 10 : 
            step_vectors = np.diff(positions, axis=0)
            step_dist = np.linalg.norm(step_vectors, axis=1)
            
            jitter_thresh = body_scale * 0.015
            step_dist[step_dist < jitter_thresh] = 0.0
            net_displacement = (np.linalg.norm(positions[-1] - positions[0]) / (body_scale + 1e-6))
            dt = (len(positions) - 1) / 30.0
            
            avg_speed = net_displacement / max(dt, 1e-6)
            path_length = np.sum(step_dist) / (body_scale + 1e-6)
            path_efficiency = (net_displacement / max(path_length, 1e-6))
            disp_score = np.clip(avg_speed * 8.0 * path_efficiency, 0.0, 1.0)
        
        scales = np.array(self.scale_history)
        if len(scales) > 5 and scales[0] > 0 and (scales[-1] / scales[0]) < 2.2 : 
            z_expansion = abs(scales[-1] - scales[0]) / (scales[0] * len(scales))
            z_score = np.clip(z_expansion * 45.0, 0.0, 1.0)
        else : z_score = 0.0
    
        
        gait_score = self.evaluate_gait(disp_score=disp_score)
        locomotion = max(disp_score, z_score)
        if gait_score > 0.4 : raw_score = 0.65 * locomotion + 0.35 * gait_score   
        else : raw_score = locomotion
        
        if person_position == "Inside" : 
            raw_score *= 0.05
            gait_score = 0.0
            disp_score *= 0.05
        
        if not self.score_history : score = raw_score
        else : score = 0.85 * self.score_history[-1] + 0.15 * raw_score
        self.score_history.append(score)
                
        transition_event = None
        if self.current_state == "STANDING" :
            if score >= self.tau_enter : 
                self.current_state = "WALKING"
                transition_event = "ENTERED_WALKING"
        elif self.current_state == "WALKING" : 
            if score <= self.tau_exit : 
                self.current_state = "STANDING"
                transition_event = "EXITED_WALKING"
        
        if person_position == "Inside" : 
            self.score_history.clear()
            self.score_history.append(0.0)
            gait_score,disp_score = 0.0, 0.0
            transition_event = None
            
            if self.current_state == "WALKING":
                self.current_state = "STANDING"
                transition_event = "EXITED_WALKING" 
                
            return self.current_state, 0.0, transition_event, 0.0, 0.0
        
        return self.current_state, score, transition_event, disp_score, gait_score

KEYPOINT_CONNECTIONS = [("l_shoulder", "r_shoulder"),("l_shoulder", "l_hip"),("r_shoulder", "r_hip"),("l_hip", "r_hip"),("l_hip", "l_knee"),("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle")]
   
def draw_keypoints(frame, keypoints, color=(0, 255, 255)) : 
    if not keypoints : return
    
    for _, point in keypoints.items() :
        pt = (int(point[0]), int(point[1]))
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)
    
    for kp1, kp2 in KEYPOINT_CONNECTIONS : 
        if kp1 in keypoints and kp2 in keypoints : cv2.line(frame, (int(keypoints[kp1][0]), int(keypoints[kp1][1])), (int(keypoints[kp2][0]), int(keypoints[kp2][1])), color, 2)
            
def transfer_state(old_id, new_id, movement_estimators, active_walkers, kalman_tracker, cross_cut_tracker=None, is_shot_cut=False):
    if old_id == new_id : return 
    
    if old_id in movement_estimators : 
        old_est = movement_estimators.pop(old_id)
        if is_shot_cut : old_est.reset()
        if new_id not in movement_estimators or len(old_est.pos_history) > len(movement_estimators[new_id].pos_history): movement_estimators[new_id] = old_est
    
    if old_id in active_walkers : 
        old_w = active_walkers.pop(old_id)
        if new_id not in active_walkers : active_walkers[new_id] = old_w
        else : 
            active_walkers[new_id] = {
                "start_time" : min(old_w["start_time"], active_walkers[new_id]["start_time"]),
                "last_seen_frame" : max(old_w["last_seen_frame"], active_walkers[new_id]["last_seen_frame"])
            }
        
    if kalman_tracker is not None and old_id in kalman_tracker : kalman_tracker[new_id] = kalman_tracker.pop(old_id)
    if cross_cut_tracker is not None and old_id in cross_cut_tracker.active_gallery : cross_cut_tracker.active_gallery[new_id] = cross_cut_tracker.active_gallery.pop(old_id)
        
def main_process(filename, person_model, car_model, frame_callback=None, out_writer=None, cue_file=None, track_counter=None):
    cap = cv2.VideoCapture(filename=filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps) : fps = 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    own_writer = False
    if out_writer is None :
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter("annotated.mp4", fourcc, fps, (width, height))
        own_writer = True
        
    shot_detector = ShotDetector(threshold=0.35, b_thresh=0.55, cooldown_frames=15)
    cross_cut_tracker = CrossCutTracker(match_thresh=0.70)
    car_verifier = InCarVisualVerifier()
    char_tracker = CharacterTracker()
    bg_est = BackgroundMotion()
    
    track_to_char_map, movement_estimators, kalman_tracker, active_walkers = {}, {}, {}, {}
    if track_counter is None : track_counter = [1]
    vehicle_results = None
    frame_since_cut = 999
    dx, dy = 0.0, 0.0
    frame_count = 0 
    
    own_cue = False
    if cue_file is None :
        cue_file = open("Walkers.cue", "w")
        cue_file.write('TITLE "Movement and Character Motion Markers"\n')
        cue_file.write('FILE "color_event.mp4" MP4\n')
        own_cue = True
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret : break
        frame_count += 1        
        relative_time = frame_count / fps
        abs_time = START_TIME_OFFSET + relative_time
        
        is_shot_cut = shot_detector.detect_cut(frame)
        cut_time = START_TIME_OFFSET + ((frame_count - 1) / fps)
        if is_shot_cut : 
            print(f"[SHOT CUT] Frame {frame_count} - Clearing motion history & Kalman filters.")
            bg_est.prev_gray = None
            
            for char_id, info in list(active_walkers.items()) :
                start_time = info["start_time"]
                if cut_time - start_time > 0.3 : 
                    write_cue(cue_file, track_counter[0], char_id, start_time, cut_time, fps)
                    track_counter[0] += 1
                
            for m_est in movement_estimators.values() : m_est.reset()
            kalman_tracker.clear()
            char_tracker.reset()
            active_walkers.clear()
            frame_since_cut = 0
        else : frame_since_cut += 1
            
        in_shot_grace = frame_since_cut < 15
        pose_results = person_model(frame, verbose=False)[0]
        if frame_count % 15 == 0 : vehicle_results = car_model(frame, classes=[2], verbose=False)[0]
        person_boxes, car_boxes = [], []
        if pose_results.boxes is not None and pose_results.keypoints is not None :
            boxes_data = pose_results.boxes.xyxy.cpu().numpy()
            confs_data = pose_results.boxes.conf.cpu().numpy()
            kps_data = pose_results.keypoints.data.cpu().numpy()
            
            for box, conf, kps in zip(boxes_data, confs_data, kps_data):
                person_boxes.append({"box" : [float(box[0]), float(box[1]), float(box[2]), float(box[3])], "score" : float(conf), "kps" : kps})
                
        if vehicle_results.boxes is not None : 
            boxes = vehicle_results.boxes.xyxy.cpu().numpy()
            classes = vehicle_results.boxes.cls.cpu().numpy()
            confs = vehicle_results.boxes.conf.cpu().numpy()
          
            for box, cls, conf in zip(boxes, classes, confs) : 
                if int(cls) != 2 : continue
                car_boxes.append({"box" : [float(box[0]), float(box[1]), float(box[2]), float(box[3])], "score" : float(conf)})
                
        person_boxes_raw = [[*p["box"], p["score"], 0] for p in person_boxes]  
        tracks = char_tracker.process(frame, frame_count, person_boxes_raw)
        all_boxes = [track['box'] for track in tracks]
        
        if frame_count % 5 == 0 : dx, dy = bg_est.estimate_shift(frame, all_boxes)
        for _, kf in kalman_tracker.items() : kf.predict(camera_shift=(dx, dy))

        process_chars_this_frame = set()        
        occluded_tracks = set()
        for i, t1 in enumerate(tracks) : 
            for j, t2 in enumerate(tracks):
                if i < j : 
                    if computer_iou(t1["box"], t2["box"]) > 0.35 :
                        occluded_tracks.add(t1["track_id"])
                        occluded_tracks.add(t2["track_id"])
        
        matched_pose_indices = set()
        track_to_pose = {}
        for track in tracks : 
            t_box = track['box']
            t_id = track['track_id']
            is_curr_occluded = t_id in occluded_tracks
            if t_id in occluded_tracks : continue
            
            best_iou = 0.40
            best_idx = None
            for p_idx, pose in enumerate(person_boxes) : 
                if p_idx in matched_pose_indices : continue
                iou = computer_iou(t_box, pose['box'])
                if iou > best_iou : 
                    best_iou = iou
                    best_idx = p_idx
                    
            if best_idx is not None : 
                matched_pose_indices.add(best_idx)
                kps_dict, is_valid_gait = extract_keypoints(person_boxes[best_idx]["kps"], frame_shape=frame.shape)
                track_to_pose[t_id] = (kps_dict, is_valid_gait)
                
        assigned_in_frame = set()
        for t in tracks:
            c_id = t.get('char_id')
            if c_id and not c_id.startswith("Track") and c_id != "Unassigned" : assigned_in_frame.add(c_id)
        
        for car in car_boxes : 
            x1, y1, x2, y2 = map(int, car["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"CAR {car['score']:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
        
        for track in tracks:
            track_id = track['track_id']
            char_id = track.get('char_id', f"Person_{track_id}")
            box = track['box']
            
            if char_id in process_chars_this_frame : continue
            process_chars_this_frame.add(char_id)
            
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(width, int(box[2])), min(height, int(box[3]))
            person_crop = frame[y1:y2, x1:x2]
            
            person_location = "Outside"
            if person_crop.size > 0 : person_location = car_verifier.verify(person_crop)
            
            if char_id.startswith("Track") or char_id == "Unassigned" : 
                matched_reid = cross_cut_tracker.find_match(person_crop, box, (dx, dy), frame.shape, exclude_ids=assigned_in_frame, kalman_tracker=kalman_tracker, is_shot_cut=in_shot_grace)
                if matched_reid and matched_reid != char_id : 
                    old_temp = char_id
                    char_id = matched_reid
                    transfer_state(old_temp, char_id, movement_estimators, active_walkers, kalman_tracker, is_shot_cut=in_shot_grace)
                    if char_id in active_walkers and char_id in movement_estimators : pass

            if not char_id.startswith("Track") and char_id != "Unassigned" : assigned_in_frame.add(char_id)
            
            prev_char_id = track_to_char_map.get(track_id)
            track_to_char_map[track_id] = char_id
            
            if char_id not in movement_estimators : movement_estimators[char_id] = MovementTracker()
            is_near_edge = movement_estimators[char_id].is_near_edge(box, frame.shape)
            is_curr_occluded = track_id in occluded_tracks
            
            if char_id not in kalman_tracker : kalman_tracker[char_id] = KalmanBoxTracker(box)
            
            if is_curr_occluded : box = kalman_tracker[char_id].get_box()
            else : box = kalman_tracker[char_id].update(box)
                        
            if not is_curr_occluded and not is_near_edge : 
                if frame_count % 15 == 0 : cross_cut_tracker.update(char_id, person_crop, box, velocity=(dx, dy))
            
            if prev_char_id is not None and prev_char_id != char_id: transfer_state(prev_char_id, char_id, movement_estimators, active_walkers, kalman_tracker, cross_cut_tracker, is_shot_cut=in_shot_grace)
            
            matched_data = track_to_pose.get(track_id) if not is_curr_occluded else None
            matched_kps_dict, is_valid_gait = matched_data if matched_data else (None, False)
            if person_location == "Inside" : is_valid_gait = False                            
            state, score, event, disp_score, gait_score = movement_estimators[char_id].update(box, frame.shape, (dx, dy), frame_count, keypoints=matched_kps_dict, is_occluded=is_curr_occluded, char_id=char_id, is_shot_cut=in_shot_grace, is_valid_gait=is_valid_gait, person_position=person_location)
            
            if char_id in active_walkers : active_walkers[char_id]["last_seen_frame"] = frame_count
            
            if event == "ENTERED_WALKING" and not char_id.startswith("Track") and char_id != "Unassigned": 
                if char_id not in active_walkers :
                    estimated_start = abs_time - (15 / fps)
                    if (estimated_start - START_TIME_OFFSET) < START_PADDING : effective_start_time = START_TIME_OFFSET
                    else : effective_start_time = estimated_start
                    active_walkers[char_id] = { "start_time" : effective_start_time, "last_seen_frame" : frame_count,}
                    print(f"[{relative_time:.4f}s] {char_id} Track {track_id} STARTED walking (Score: {score:.2f})")
            elif event == "EXITED_WALKING" and char_id in active_walkers: 
                walker_info = active_walkers.pop(char_id)
                start_time = walker_info["start_time"]
                end_time = abs_time
                write_cue(cue_file, track_counter[0], char_id, start_time, end_time, fps)
                track_counter[0] += 1
                print(f"[{relative_time:.4f}s] {char_id} Track {track_id} STOPPED walking (Score: {score:.2f})")
        
            if state == "WALKING" :color = (0, 255, 0) 
            elif state == "STANDING" : color = (0, 165, 255)
            else : color = (255, 255, 0)
            
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, p1, p2, color, 2)
            label_text = f"{char_id} (Track {track_id} - {person_location}) : {state} | Score : {score:.3f} Gait : {gait_score:.3f}  Displacement : {disp_score:.3f}"
            cv2.putText(frame, label_text, (p1[0], max(15, p1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            visual_kps = movement_estimators[char_id].smoothed_kps or matched_kps_dict
            if visual_kps : draw_keypoints(frame, visual_kps, color=color)
            
        missing_chars = []
        for char_id, info in active_walkers.items() : 
            if frame_count - info["last_seen_frame"] >= 24 : missing_chars.append(char_id)    
        
        for char_id in missing_chars : 
            info = active_walkers.pop(char_id)
            start_time = info["start_time"]
            end_time = START_TIME_OFFSET + (info["last_seen_frame"] / fps)
            if end_time - start_time > 0.3 : 
                write_cue(cue_file, track_counter[0], char_id, start_time, end_time, fps)
                track_counter[0] += 1
            
            if char_id in movement_estimators : movement_estimators[char_id].reset()
            if char_id in kalman_tracker : del kalman_tracker[char_id]

        if frame_callback is not None : frame_callback(frame)

        out_writer.write(frame)
    
    for char_id, info in active_walkers.items() : 
        start_time = info["start_time"]
        end_time = START_TIME_OFFSET + (info["last_seen_frame"] / fps)
        if end_time - start_time > 0.3 :
            write_cue(cue_file, track_counter[0], char_id, start_time, cut_time, fps)
            track_counter[0] += 1
    
    if own_cue : cue_file.close()
    cap.release()
    if own_writer : out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" : 
    pose_model = YOLO("YoloModels/yolo11n-pose.pt")
    vehicle_model = YOLO("YoloModels/yolo11n.pt")   
    main_process("Media/color_event_48.mp4", pose_model, vehicle_model)
    
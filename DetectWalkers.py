import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from collections import deque
from ultralytics import YOLO
from TrackFaces import CharacterTracker

pose_model = YOLO("yolo11n-pose.pt")
START_TIME_OFFSET = 446 #314 or 3028 or 329 or 446
START_PADDING = 2.0

def computer_iou(box1, box2) : 
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(b1_area + b2_area - inter_area + 1e-6)

def extract_keypoints(kps, conf_thresh=0.15):
    indices = [11,12,13,14,15,16,5,6]
    if len(kps) < 17 : return None
    
    extracted = {}
    key_names = ["l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle", "l_shoulder", "r_shoulder"]
    for idx, name in zip(indices, key_names) : 
        kp = kps[idx]
        if len(kp) >= 3 and kp[2] >= conf_thresh : extracted[name] = kp[:2]
    
    has_shoulders = {"l_shoulder", "r_shoulder"}.issubset(extracted.keys())
    has_legs = {"l_hip", "l_knee", "l_ankle"}.issubset(extracted.keys()) or {"r_hip", "r_knee", "r_ankle"}.issubset(extracted.keys())
    has_knees = {"l_knee", "r_knee"}.issubset(extracted.keys())
    has_ankles = {"l_ankle", "r_ankle"}.issubset(extracted.keys())
    return extracted if (has_shoulders or has_legs or has_knees or has_ankles) else None

class AppearenceReID : 
    def __init__(self) : 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        
        self.feature_extractor = torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d((1, 1)))
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extraction(self, crop) : 
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10 : return None
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        features = self.feature_extractor(input_tensor).flatten(1)
        features = features.squeeze(0)
        
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
                updated_emb = 0.95 * old_emb + 0.05 * emb
                updated_emb = updated_emb / (np.linalg.norm(updated_emb) + 1e-6)
                self.active_gallery[track_id] = {"embedding" : updated_emb, "last_pos" : centroid, "velocity" : velocity}
            else : self.active_gallery[track_id] = {"embedding" : emb, "last_pos" : centroid, "velocity" : velocity}
                
    def find_match(self, new_crop, new_box, camera_shift, frame_shape, exclude_ids) : 
        new_emb = self.reid.extraction(new_crop)
        if new_emb is None : return None
        
        h_frame, w_frame = frame_shape[:2]
        new_centroid = np.array([(new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0])
        scores = []
        
        for track_id, data in self.active_gallery.items():
            sim = self.reid.compare_embeddings(new_emb, data["embedding"])
            if sim < 0.70 or track_id in exclude_ids: continue
            
            predicted_pos = data['last_pos'] - np.array(camera_shift)
            dist = np.linalg.norm(new_centroid - predicted_pos)
            max_diag = np.sqrt(w_frame**2 + h_frame**2)
            s_pos = max(0.0, 1.0 - (dist / (max_diag * 0.2)))
            
            shift_diff = np.linalg.norm(np.array(camera_shift) - np.array(data["velocity"]))
            s_shift = max(0.0, 1.0 - (shift_diff / 50.0))
            
            score = (self.w_appearance * sim + self.w_position * s_pos + self.w_velocity * s_shift)
            scores.append((score, track_id))
         
        if not scores : return None
        
        scores.sort(key=lambda x: x[0], reverse=True) 
        best_score, best_id = scores[0]
        if best_score < self.match_thresh : return None
        
        if len(scores) > 1 : 
            second_best = scores[1][0]
            if (best_score - second_best) < 0.10 : return None
                  
        return best_id
    
class BackgroundMotion:
    def __init__(self) : self.prev_gray = None
   
    def estimate_shift(self, frame, person_boxes) :
        if len(frame.shape) == 3 : current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else : current_gray = frame.copy()
        
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
        mean_flow = cv2.mean(flow, mask=bg_mask)
        return float(mean_flow[0]), float(mean_flow[1])
        
class MovementTracker:
    def __init__(self, buffer_size=40, tau_enter=0.25, tau_exit=0.05):
        self.buffer_size = buffer_size
        self.tau_enter = tau_enter
        self.tau_exit = tau_exit
        
        self.pos_history = deque(maxlen=buffer_size)
        self.gait_history = deque(maxlen=buffer_size)
        self.height_history = deque(maxlen=buffer_size)
        self.shoulder_history = deque(maxlen=buffer_size)
        self.smoothed_kps = {}
        self.current_state = "STANDING"
        self.walking = False
        
        self.smoothed_scale = None
        self.last_centroid = None
        self.last_frame_seen = -1
        self.current_position = None
    
    def is_near_edge(self, box, frame_shape, margin=15):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        return x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >=(h - margin)
    
    def calc_angle(self, p1, p2, p3):
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
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
        self.pos_history.clear()
        self.gait_history.clear()
        self.height_history.clear()
        self.shoulder_history.clear()
        self.smoothed_kps.clear()
        self.current_state = "STANDING"
        self.last_centroid = None
        self.smoothed_scale = None
        self.last_frame_seen = -1
        self.current_position = None
    
    def update(self, box, frame_shape, camera_shift, frame_idx, keypoints=None, is_occluded=False, char_id = None):
        h_frame, w_frame = frame_shape[:2]
        if (h_frame == 0 or w_frame == 0) and self.current_state != "WALKING": return "ANALYZING", 0.0, None, 0.0, 0.0
        
        if len(self.pos_history) == 0 and self.is_near_edge(box, frame_shape) : return self.current_state, 0.0, None, 0.0, 0.0
        
        cam_dx, cam_dy = camera_shift
        (cx, cy), body_scale = self.get_center_and_scale(keypoints, box)
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
        self.last_centroid = (cx, cy)
        self.last_frame_seen = frame_idx
        
        has_legs = False
        if keypoints:
            alpha = 0.35
            smoothed = {}
            for name, pt in keypoints.items() :
                if name in self.smoothed_kps : smoothed[name] = alpha * np.array(pt) + (1.0 - alpha) * self.smoothed_kps[name]
                else : smoothed[name] = np.array(pt)
            
            self.smoothed_kps = smoothed
            kps = self.smoothed_kps
            
            if "l_ankle" in kps and "r_ankle" in kps :
                dist = np.linalg.norm(np.array(kps["l_ankle"]) - np.array(kps["r_ankle"])) / body_scale
                self.gait_history.append(dist)
                has_legs = True
            elif "l_knee" in kps and "r_knee" in kps :
                dist = np.linalg.norm(np.array(kps["l_knee"]) - np.array(kps["r_knee"])) / body_scale
                self.gait_history.append(dist)
                has_legs = True
            if "l_shoulder" in kps and "r_shoulder" in kps :
                sh_dist = np.linalg.norm(np.array(kps["l_shoulder"]) - np.array(kps["r_shoulder"])) / body_scale
                self.shoulder_history.append(sh_dist)
                
        if len(self.pos_history) < 15 and self.current_state != "WALKING": return "ANALYZING", 0.0, None, 0.0, 0.0
        
        positions = np.array(self.pos_history)
        step_vectors = np.diff(positions, axis=0)
        total_path_length = np.sum(np.linalg.norm(step_vectors, axis=1)) / body_scale
        disp_score = total_path_length / (len(self.pos_history) * 0.15) 
        shoulder_var = 0.0
        height_var = 0.0
        if not has_legs : 
            if len(self.shoulder_history) >= 4 : 
                shoulder_std = float(np.std(self.shoulder_history))
                shoulder_mean = float(np.mean(self.shoulder_history))
                shoulder_var = shoulder_std / (shoulder_mean + 1e-6)
            if len(self.height_history) >= 4 :
                height_std = float(np.std(self.height_history))
                height_mean = float(np.mean(self.height_history))
                height_var = height_std / (height_mean + 1e-6)
            
            shoulder_bool = (len(self.shoulder_history) < 4) or (shoulder_var < 0.08)
            height_bool = (len(self.height_history) < 4) or (height_var < 0.04)
            
            if shoulder_bool and height_bool : disp_score *= 0.05
            elif shoulder_bool or height_bool : disp_score *= 0.20
            
        gait_score = 0.0
        if len(self.gait_history) >= 4 and disp_score > 0.1:
            raw_gait = float(np.std(self.gait_history)) 
            gait_score = max(0.0, raw_gait - 0.012) * 10.0
        
        print( char_id, f"disp={disp_score:.3f}", f"gait={gait_score:.3f}", f"shoulder={shoulder_var:.3f}", f"height={height_var:.3f}")
        
        score = max(disp_score, gait_score)
            
        transition_event = None
        if self.current_state == "STANDING" :
            if score >= self.tau_enter : 
                self.current_state = "WALKING"
                transition_event = "ENTERED_WALKING"
        elif self.current_state == "WALKING" : 
            if score <= self.tau_exit : 
                self.current_state = "STANDING"
                transition_event = "EXITED_WALKING"
        
        return self.current_state, score, transition_event, disp_score, gait_score

KEYPOINT_CONNECTIONS = [
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_hip", "r_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
]
   
def draw_keypoints(frame, keypoints, color=(0, 255, 255)) : 
    if not keypoints : return
    
    for name, point in keypoints.items() :
        pt = (int(point[0]), int(point[1]))
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)
    
    for kp1, kp2 in KEYPOINT_CONNECTIONS : 
        if kp1 in keypoints and kp2 in keypoints : 
            pt1 = (int(keypoints[kp1][0]), int(keypoints[kp1][1]))
            pt2 = (int(keypoints[kp2][0]), int(keypoints[kp2][1]))
            cv2.line(frame, pt1, pt2, color, 2)
            
def transfer_state(old_id, new_id, movement_estimators, active_walkers):
    if old_id == new_id : return
    
    if old_id in movement_estimators : 
        if new_id not in movement_estimators or movement_estimators[old_id].current_state == "WALKING" or len(movement_estimators[old_id].pos_history) > len(movement_estimators[new_id].pos_history): movement_estimators[new_id] = movement_estimators[old_id]
        del movement_estimators[old_id]
    
    if old_id in active_walkers : 
        if new_id not in active_walkers : active_walkers[new_id] = active_walkers[old_id]
        else : 
            earliest_start = min(active_walkers[old_id]["start_time"], active_walkers[new_id]["start_time"])
            latest_frame = max(active_walkers[old_id]["last_seen_frame"], active_walkers[new_id]["last_seen_frame"])
            active_walkers[new_id] = {"start_time" : earliest_start, "last_seen_frame" : latest_frame}
        del active_walkers[old_id]
        
def main():
    cap = cv2.VideoCapture("Media/color_event_48.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps) : fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter("annotated.mp4", fourcc, fps, (width, height))
    
    char_tracker = CharacterTracker()
    cross_cut_tracker = CrossCutTracker(match_thresh=0.70)
    bg_est = BackgroundMotion()
    track_to_char_map = {}
    movement_estimators = {}
    frame_count = 0
    
    txt_file = open("Walkers.txt", "w")
    txt_file.write("Actor_ID, Start Time, End Time, Duration\n")
    active_walkers = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret : break
        frame_count += 1        
        relative_time = frame_count / fps
        abs_time = START_TIME_OFFSET + relative_time
        
        pose_results = pose_model(frame, verbose=False)[0]
        person_boxes = []
        if pose_results.boxes is not None and pose_results.keypoints is not None :
            boxes_data = pose_results.boxes.xyxy.cpu().numpy()
            confs_data = pose_results.boxes.conf.cpu().numpy()
            kps_data = pose_results.keypoints.data.cpu().numpy()
            
            for box, conf, kps in zip(boxes_data, confs_data, kps_data):
                person_boxes.append({
                    "box" : [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "score" : float(conf),
                    "kps" : kps
                })
          
        person_boxes_raw = [[*p["box"], p["score"], 0] for p in person_boxes]  
        tracks = char_tracker.process(frame, frame_count, person_boxes_raw)
        all_boxes = [track['box'] for track in tracks]
        
        dx, dy = bg_est.estimate_shift(frame, all_boxes)

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
                track_to_pose[t_id] = extract_keypoints(person_boxes[best_idx]["kps"])
                
        assigned_in_frame = set()
        for t in tracks:
            c_id = t.get('char_id')
            if c_id and not c_id.startswith("Track") and c_id != "Unassigned" : assigned_in_frame.add(c_id)
            
        for track in tracks:
            track_id = track['track_id']
            char_id = track.get('char_id', f"Person_{track_id}")
            box = track['box']
            
            if char_id in process_chars_this_frame : continue
            process_chars_this_frame.add(char_id)
            
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(width, int(box[2])), min(height, int(box[3]))
            person_crop = frame[y1:y2, x1:x2]
            
            if char_id.startswith("Track") or char_id == "Unassigned" : 
                matched_reid = cross_cut_tracker.find_match(person_crop, box, (dx, dy), frame.shape, exclude_ids=assigned_in_frame)
                if matched_reid and matched_reid != char_id : 
                    old_temp = char_id
                    char_id = matched_reid
                    #print(f"[REID] Track {track_id} | " f"{old_temp if 'old_temp' in locals() else char_id} -> {char_id} | " f"movement states={list(movement_estimators.keys())}")
                    transfer_state(old_temp, char_id, movement_estimators, active_walkers)
                    if char_id in active_walkers and char_id in movement_estimators : movement_estimators[char_id].current_state = "WALKING"

            if not char_id.startswith("Track") and char_id != "Unassigned" : assigned_in_frame.add(char_id)
            
            prev_char_id = track_to_char_map.get(track_id)
            track_to_char_map[track_id] = char_id
            
            if char_id not in movement_estimators : movement_estimators[char_id] = MovementTracker()
            is_near_edge = movement_estimators[char_id].is_near_edge(box, frame.shape)
            is_curr_occluded = track_id in occluded_tracks
            if not char_id.startswith("Track") and char_id != "Unassigned" and not is_curr_occluded and not is_near_edge : cross_cut_tracker.update(char_id, person_crop, box, velocity=(dx, dy))   
            if prev_char_id is not None and prev_char_id != char_id: transfer_state(prev_char_id, char_id, movement_estimators, active_walkers)
            
            matched_kps_dict = track_to_pose.get(track_id) if not is_curr_occluded else None                            
            state, score, event, disp_score, gait_score = movement_estimators[char_id].update(box, frame.shape, (dx, dy), frame_count, keypoints=matched_kps_dict, is_occluded=is_curr_occluded, char_id=char_id)
            #print(char_id, state, f"score={score:.3f}", f"disp={disp_score:.3f}", f"gait={gait_score:.3f} occluded={is_curr_occluded}", f"history={len(movement_estimators[char_id].pos_history)}")
            
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
                duration = end_time - start_time
                txt_file.write(f"{char_id}, {start_time:.2f}, {end_time:.2f}, {duration:.2f}\n")
                txt_file.flush()
                print(f"[{relative_time:.4f}s] {char_id} Track {track_id} STOPPED walking (Score: {score:.2f})")
        
            if state == "WALKING" :color = (0, 255, 0) 
            elif state == "STANDING" : color = (0, 165, 255)
            else : color = (255, 255, 0)
            
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            label_text = f"{char_id} (Track {track_id}) : {state} | Score : {score:.3f}"
            cv2.putText(frame, label_text, (box[0], max(15, box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            visual_kps = movement_estimators[char_id].smoothed_kps or matched_kps_dict
            if visual_kps : draw_keypoints(frame, visual_kps, color=color)
            
        missing_chars = []
        for char_id, info in active_walkers.items() : 
            if frame_count - info["last_seen_frame"] >= 24 : missing_chars.append(char_id)    
        
        for char_id in missing_chars : 
            info = active_walkers.pop(char_id)
            start_time = info["start_time"]
            end_time = START_TIME_OFFSET + (info["last_seen_frame"] / fps)
            txt_file.write(f"{char_id}, {start_time:.2f}, {end_time:.2f}, {(end_time - start_time):.2f}\n")
            txt_file.flush()
            
            if char_id in movement_estimators : movement_estimators[char_id].reset()

        out_writer.write(frame)
        cv2.imshow("Movement", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') : break
    
    for char_id, info in active_walkers.items() : 
        start_time = info["start_time"]
        end_time = START_TIME_OFFSET + (info["last_seen_frame"] / fps)
        txt_file.write(f"{char_id}, {start_time:.2f}, {end_time:.2f}, {(end_time - start_time):.2f}\n")
    
    txt_file.close()
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" : main()
    
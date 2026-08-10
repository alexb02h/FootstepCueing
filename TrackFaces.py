import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.engine.results import Boxes
from types import SimpleNamespace


class CharacterTracker:
    def __init__(self, similarity_thresh=0.48, yaw_thresh=30.0):
        self.app = FaceAnalysis(name='buffalo_sc', providers=['CoreMLExecutionProvider','CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        tracker_args = SimpleNamespace(
            track_high_thresh = 0.5,
            track_low_thresh = 0.1,
            new_track_thresh = 0.6,
            track_buffer = 30,
            match_thresh = 0.5,
            gmc_method = 'sparseOptFlow',
            proximity_thresh = 0.5,
            appearance_thresh = 0.25,
            with_reid = False,
            model = 'yolov8n-cls.pt',
            device = 'cpu',
            fuse_score = True,
            frame_rate = 30
        )
        self.bot_tracker = BOTSORT(args=tracker_args)

        self.characters = {}
        self.next_character_id = 1
        self.track_to_char_map = {}
        self.track_confirmed = {}

        self.similarity_thresh = similarity_thresh
        self.yaw_thresh = yaw_thresh

    def yaw(self, kps):
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        dist_l = abs(nose[0] - left_eye[0])
        dist_r = abs(right_eye[0] - nose[0])
    
        if dist_l + dist_r == 0 : return 0.0
    
        ratio = (dist_l - dist_r) / (dist_l + dist_r)
        return abs(ratio * 90.0)

    def get_character_id(self, face_embedding, yaw_angle):
        norm_embed = face_embedding / np.linalg.norm(face_embedding)
        best_match = None
        highest_similarity = -1.0
    
        for char_id, embeding_list in self.characters.items():
            sims = np.dot(embeding_list, norm_embed)
            top_k = min(3, len(sims))
            top_sims = np.partition(sims, -top_k)[-top_k:]
            avg_top_sims = np.mean(top_sims)
        
            if avg_top_sims > highest_similarity:
                highest_similarity = avg_top_sims
                best_match = char_id
            
        if highest_similarity > self.similarity_thresh and best_match is not None:
            if yaw_angle < self.yaw_thresh:
                if len(self.characters[best_match]) >= 30: self.characters[best_match].pop(0)
                self.characters[best_match].append(norm_embed)
            return best_match, True
        
        if best_match is not None and highest_similarity > self.similarity_thresh - 0.15 : return None, False
        
        elif yaw_angle < self.yaw_thresh:
            new_id = f"Character_{self.next_character_id}"
            self.characters[new_id] = [norm_embed]
            self.next_character_id += 1
            return new_id, False
        else: return None, False
    
    def process(self, frame, frame_count, person_boxes):
        results = []
        if len(person_boxes) == 0 : return results
        
        det_tensor = np.array(person_boxes)
        if det_tensor.shape[1] == 5 : det_tensor = np.hstack([det_tensor, np.zeros((len(det_tensor), 1))])
        
        boxes = Boxes(det_tensor, orig_shape=frame.shape[:2])
        tracks = self.bot_tracker.update(boxes, frame)
        faces = self.app.get(frame)
        valid_faces = [f for f in faces if f.det_score >= 0.6 and (f.bbox[2] - f.bbox[0]) >= 30]
            
        for track in tracks : 
            if hasattr(track, 'tlbr') : 
                t_box = track.tlbr.astype(int)
                track_id = int(track.track_id)
            else : 
                t_box = track[:4].astype(int)
                track_id = int(track[4])
                
            is_confirmed = self.track_confirmed.get(track_id, False)
            if not is_confirmed or frame_count % 5 == 0 :
                matched_face = None
                for face in valid_faces : 
                    fb = face.bbox
                    face_cx = (fb[0] + fb[2]) / 2.0
                    face_cy = (fb[1] + fb[3]) / 2.0
                    
                    if (t_box[0] <= face_cx <= t_box[2]) and (t_box[1] <= face_cy <= t_box[3]) : 
                        matched_face = face
                        break
                
                if matched_face is not None :
                    yaw_angle = self.yaw(matched_face.kps)
                    char_id, confirmed = self.get_character_id(matched_face.embedding, yaw_angle)
                    
                    if char_id is not None :
                        self.track_to_char_map[track_id] = char_id
                        self.track_confirmed[track_id] = confirmed
                        
            char_label = self.track_to_char_map.get(track_id, f"Track_{track_id}")
            results.append({'track_id': track_id, 'char_id': char_label, 'box': t_box})
            
        return results